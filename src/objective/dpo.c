// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#include "slate/dpo.h"
#include "slate/tensor.h"
#include "slate/autograd.h"
#include "slate/runtime.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    float* softmax_cache;
    int32_t* tgt_cache;
    int64_t B, T, V;
} seq_logp_state_t;

static void seq_logp_backward(slate_graph_node_t* node) {
    slate_tensor_t* logits = node->inputs[0];
    slate_tensor_t* out = node->output;
    if (!logits->requires_grad || !logits->grad) return;
    seq_logp_state_t* st = (seq_logp_state_t*)node->user_data;
    const float* dy = (const float*)out->grad;
    float* dl = (float*)logits->grad;
    for (int64_t b = 0; b < st->B; ++b) {
        float gb = dy[b];
        for (int64_t t = 0; t < st->T; ++t) {
            const float* sm = st->softmax_cache + (b * st->T + t) * st->V;
            float* dl_row = dl + (b * st->T + t) * st->V;
            int32_t tgt = st->tgt_cache[b * st->T + t];
            for (int64_t v = 0; v < st->V; ++v) {
                float ind = (v == tgt) ? 1.0f : 0.0f;
                dl_row[v] += gb * (ind - sm[v]);
            }
        }
    }
}

slate_tensor_t* slate_op_seq_logp(slate_graph_ctx_t* ctx,
                                   slate_tensor_t* logits, slate_tensor_t* targets) {
    if (!ctx || !logits || !targets) return NULL;
    if (logits->dtype != SLATE_DTYPE_F32 || targets->dtype != SLATE_DTYPE_I32) return NULL;
    if (logits->n_dims != 3 || targets->n_dims != 2) return NULL;
    int64_t B = logits->shape[0], T = logits->shape[1], V = logits->shape[2];
    if (targets->shape[0] != B || targets->shape[1] != T) return NULL;
    int64_t out_shape[1] = {B};
    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32, 1, out_shape, false);
    float* sm = (float*)slate_arena_alloc(ctx->scratch_arena, (size_t)(B*T*V)*sizeof(float), 16);
    const float* pl = (const float*)logits->data;
    const int32_t* pt = (const int32_t*)targets->data;
    float* po = (float*)out->data;
    for (int64_t b = 0; b < B; ++b) {
        double total_lp = 0.0;
        for (int64_t t = 0; t < T; ++t) {
            const float* row = pl + (b * T + t) * V;
            float* srow = sm + (b * T + t) * V;
            float m = row[0];
            for (int64_t v = 1; v < V; ++v) if (row[v] > m) m = row[v];
            double S = 0.0;
            for (int64_t v = 0; v < V; ++v) { srow[v] = expf(row[v] - m); S += srow[v]; }
            float log_S = logf((float)S) + m;
            float inv_S = (float)(1.0 / S);
            for (int64_t v = 0; v < V; ++v) srow[v] *= inv_S;
            int32_t tgt = pt[b * T + t];
            total_lp += (double)(row[tgt] - log_S);
        }
        po[b] = (float)total_lp;
    }
    slate_tensor_t* inputs[2] = {logits, targets};
    slate_graph_node_t* node = slate_graph_record(ctx, "seq_logp", inputs, 2, out, seq_logp_backward);
    if (node) {
        seq_logp_state_t* st = (seq_logp_state_t*)slate_arena_alloc(ctx->scratch_arena, sizeof(*st), 16);
        int32_t* tcopy = (int32_t*)slate_arena_alloc(ctx->scratch_arena, (size_t)(B*T)*sizeof(int32_t), 16);
        memcpy(tcopy, pt, (size_t)(B*T)*sizeof(int32_t));
        st->softmax_cache = sm; st->tgt_cache = tcopy;
        st->B = B; st->T = T; st->V = V;
        node->user_data = st;
        if (!out->grad) out->grad = slate_arena_alloc(ctx->scratch_arena, (size_t)B*sizeof(float), 16);
    }
    return out;
}

typedef struct { float* delta; float beta; int64_t B; } dpo_state_t;

static void dpo_backward(slate_graph_node_t* node) {
    slate_tensor_t* logp_c = node->inputs[0];
    slate_tensor_t* logp_r = node->inputs[1];
    slate_tensor_t* out = node->output;
    dpo_state_t* st = (dpo_state_t*)node->user_data;
    float dy = ((const float*)out->grad)[0];
    float scale = dy / (float)st->B;
    float* dc = logp_c->requires_grad ? (float*)logp_c->grad : NULL;
    float* dr = logp_r->requires_grad ? (float*)logp_r->grad : NULL;
    for (int64_t b = 0; b < st->B; ++b) {
        float d = st->delta[b];
        float sig = (d >= 0.0f) ? 1.0f/(1.0f+expf(-d)) : expf(d)/(1.0f+expf(d));
        float g = -(1.0f - sig) * scale * st->beta;
        if (dc) dc[b] +=  g;
        if (dr) dr[b] += -g;
    }
}

slate_tensor_t* slate_op_dpo_loss(slate_graph_ctx_t* ctx,
                                   slate_tensor_t* chosen_logits,
                                   slate_tensor_t* chosen_targets,
                                   slate_tensor_t* chosen_logps_ref,
                                   slate_tensor_t* rejected_logits,
                                   slate_tensor_t* rejected_targets,
                                   slate_tensor_t* rejected_logps_ref,
                                   float beta) {
    slate_tensor_t* logp_c = slate_op_seq_logp(ctx, chosen_logits, chosen_targets);
    slate_tensor_t* logp_r = slate_op_seq_logp(ctx, rejected_logits, rejected_targets);
    int64_t B = logp_c->shape[0];
    int64_t s1[1] = {1};
    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32, 1, s1, false);
    const float* pc = (const float*)logp_c->data;
    const float* pr = (const float*)logp_r->data;
    const float* pc_ref = (const float*)chosen_logps_ref->data;
    const float* pr_ref = (const float*)rejected_logps_ref->data;
    float* delta = (float*)slate_arena_alloc(ctx->scratch_arena, (size_t)B*sizeof(float), 16);
    double L = 0;
    for (int64_t b = 0; b < B; ++b) {
        delta[b] = beta * (pc[b] - pr[b] - pc_ref[b] + pr_ref[b]);
        float d = delta[b];
        float sp = (d >= 0.0f) ? logf(1.0f+expf(-d)) : (-d + logf(1.0f+expf(d)));
        L += sp;
    }
    ((float*)out->data)[0] = (float)(L / (double)B);
    slate_tensor_t* inputs[2] = {logp_c, logp_r};
    slate_graph_node_t* node = slate_graph_record(ctx, "dpo_loss", inputs, 2, out, dpo_backward);
    if (node) {
        dpo_state_t* st = (dpo_state_t*)slate_arena_alloc(ctx->scratch_arena, sizeof(*st), 16);
        st->delta = delta; st->beta = beta; st->B = B;
        node->user_data = st;
    }
    if (!out->grad) out->grad = slate_arena_alloc(ctx->scratch_arena, sizeof(float), 16);
    out->requires_grad = (node != NULL) || chosen_logits->requires_grad;
    return out;
}
