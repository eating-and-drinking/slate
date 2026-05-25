// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// cross_entropy.c — fused log_softmax + NLL loss for classification.
//
// Forward computes loss = -mean_b(log_softmax(logits[b])[targets[b]]).
// Backward is the canonical fused form:
//     d_logits[b, c] = (softmax(logits[b])[c] - 1[c == targets[b]]) / B
//
// We cache softmax(logits) in node->user_data so backward doesn't recompute.

#include "slate/ops.h"
#include "slate/tensor.h"
#include "slate/autograd.h"
#include "slate/runtime.h"

#include <math.h>
#include <string.h>

typedef struct ce_state {
    float* softmax_cache;   // [B, C], allocated on scratch arena
    int32_t* targets_cache; // [B], owned copy on scratch arena (NOT aliased)
    int64_t B;
    int64_t C;
} ce_state_t;

static void ce_backward(slate_graph_node_t* node) {
    slate_tensor_t* logits = node->inputs[0];
    slate_tensor_t* out = node->output;
    if (!logits->requires_grad || !logits->grad) return;

    ce_state_t* st = (ce_state_t*)node->user_data;
    float upstream = ((const float*)out->grad)[0];
    float scale = upstream / (float)st->B;

    float* d_logits = (float*)logits->grad;
    for (int64_t b = 0; b < st->B; ++b) {
        const float* sr = st->softmax_cache + b * st->C;
        float* dr = d_logits + b * st->C;
        int32_t t = st->targets_cache[b];
        for (int64_t c = 0; c < st->C; ++c) {
            float g = sr[c];
            if (c == t) g -= 1.0f;
            dr[c] += scale * g;
        }
    }
}

slate_tensor_t* slate_op_cross_entropy_loss(slate_graph_ctx_t* ctx,
                                             slate_tensor_t* logits,
                                             slate_tensor_t* targets) {
    if (!ctx || !logits || !targets) return NULL;
    if (logits->dtype != SLATE_DTYPE_F32) return NULL;
    if (targets->dtype != SLATE_DTYPE_I32) return NULL;
    if (logits->n_dims != 2 || targets->n_dims != 1) return NULL;
    if (logits->shape[0] != targets->shape[0]) return NULL;

    int64_t B = logits->shape[0];
    int64_t C = logits->shape[1];

    // Compute softmax(logits) and store for backward.
    float* sm = (float*)slate_arena_alloc(ctx->scratch_arena,
                                          (size_t)B * (size_t)C * sizeof(float), 16);
    const float* pl = (const float*)logits->data;
    const int32_t* pt = (const int32_t*)targets->data;
    double loss_acc = 0.0;
    for (int64_t b = 0; b < B; ++b) {
        const float* row = pl + b * C;
        float* srow = sm + b * C;
        float m = row[0];
        for (int64_t c = 1; c < C; ++c) if (row[c] > m) m = row[c];
        double sum_exp = 0.0;
        for (int64_t c = 0; c < C; ++c) {
            float e = expf(row[c] - m);
            srow[c] = e;
            sum_exp += e;
        }
        float log_sum = (float)(logf((float)sum_exp) + m);
        // Normalize softmax in place.
        float inv = (float)(1.0 / sum_exp);
        for (int64_t c = 0; c < C; ++c) srow[c] *= inv;

        int32_t tgt = pt[b];
        if (tgt < 0 || tgt >= C) return NULL;  // out-of-range target
        // NLL: loss_b = -(logits[t] - log_sum) = log_sum - logits[t]
        loss_acc += (double)(log_sum - row[tgt]);
    }

    int64_t scalar_shape[1] = {1};
    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32,
                                            1, scalar_shape, false);
    if (!out) return NULL;
    ((float*)out->data)[0] = (float)(loss_acc / (double)B);

    slate_tensor_t* inputs[2] = {logits, targets};
    slate_graph_node_t* node = slate_graph_record(ctx, "cross_entropy",
                                                   inputs, 2, out, ce_backward);
    if (node) {
        ce_state_t* st = (ce_state_t*)slate_arena_alloc(ctx->scratch_arena,
                                                        sizeof(*st), 16);
        // Deep-copy targets into scratch arena so backward doesn't depend on
        // the caller's tensor still being alive.
        int32_t* tcopy = (int32_t*)slate_arena_alloc(ctx->scratch_arena,
                                                      (size_t)B * sizeof(int32_t),
                                                      16);
        memcpy(tcopy, targets->data, (size_t)B * sizeof(int32_t));
        st->softmax_cache = sm;
        st->targets_cache = tcopy;
        st->B = B;
        st->C = C;
        node->user_data = st;
    }

    // Always allocate a grad buffer; this is a loss, the backward seed lives there.
    if (!out->grad) {
        out->grad = slate_arena_alloc(ctx->scratch_arena, sizeof(float), 16);
    }
    out->requires_grad = (node != NULL) || logits->requires_grad;
    return out;
}
