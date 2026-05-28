// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#include "slate/kd.h"
#include "slate/tensor.h"
#include "slate/autograd.h"
#include "slate/runtime.h"
#include <math.h>
#include <string.h>

// KL(P||Q) = sum_v P_v * (log P_v - log Q_v)
// With temperature T: P = softmax(teacher/T), Q = softmax(student/T)
// Multiply by T^2 (Hinton convention) so gradient magnitudes match CE scale.
//
// Backward: d/d student_logits[b,t,v] = (1/T^2) * (Q - P) * T^2 = (Q - P)
// Actually: d KL / d student_logits[v] = -(P_v - Q_v) / T (per position, before T^2 factor)
// With T^2 factor: d/d student_logits[v] = -(P_v - Q_v) * T (= (Q_v - P_v) * T)

typedef struct {
    float* p_teacher;     // [B*T, V] precomputed softmax(teacher/T)
    float* q_student;     // [B*T, V] softmax(student/T) — used in backward
    float T;
    int64_t BT, V;
} kd_state_t;

static void kd_backward(slate_graph_node_t* node) {
    slate_tensor_t* s = node->inputs[0];
    slate_tensor_t* out = node->output;
    if (!s->requires_grad || !s->grad) return;
    kd_state_t* st = (kd_state_t*)node->user_data;
    float dy = ((const float*)out->grad)[0];
    float scale = (dy / (float)st->BT) * st->T;
    float* dl = (float*)s->grad;
    for (int64_t i = 0; i < st->BT; ++i) {
        const float* p = st->p_teacher + i * st->V;
        const float* q = st->q_student + i * st->V;
        float* dr = dl + i * st->V;
        for (int64_t v = 0; v < st->V; ++v) dr[v] += scale * (q[v] - p[v]);
    }
}

slate_tensor_t* slate_op_kd_loss(slate_graph_ctx_t* ctx,
                                  slate_tensor_t* student,
                                  slate_tensor_t* teacher,
                                  float T) {
    if (!ctx || !student || !teacher) return NULL;
    if (student->n_dims != 3 || teacher->n_dims != 3) return NULL;
    int64_t B = student->shape[0], Tt = student->shape[1], V = student->shape[2];
    int64_t BT = B * Tt;
    int64_t s1[1] = {1};
    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32, 1, s1, false);
    float* p_t = (float*)slate_arena_alloc(ctx->scratch_arena, (size_t)BT*V*sizeof(float), 16);
    float* q_s = (float*)slate_arena_alloc(ctx->scratch_arena, (size_t)BT*V*sizeof(float), 16);
    const float* ps = (const float*)student->data;
    const float* pt = (const float*)teacher->data;
    double L = 0;
    for (int64_t i = 0; i < BT; ++i) {
        // softmax(student / T)
        float m_s = ps[i*V] / T;
        for (int64_t v = 1; v < V; ++v) { float x = ps[i*V+v]/T; if (x > m_s) m_s = x; }
        double Ss = 0;
        for (int64_t v = 0; v < V; ++v) { q_s[i*V+v] = expf(ps[i*V+v]/T - m_s); Ss += q_s[i*V+v]; }
        for (int64_t v = 0; v < V; ++v) q_s[i*V+v] = (float)(q_s[i*V+v] / Ss);
        // softmax(teacher / T)
        float m_t = pt[i*V] / T;
        for (int64_t v = 1; v < V; ++v) { float x = pt[i*V+v]/T; if (x > m_t) m_t = x; }
        double St = 0;
        for (int64_t v = 0; v < V; ++v) { p_t[i*V+v] = expf(pt[i*V+v]/T - m_t); St += p_t[i*V+v]; }
        for (int64_t v = 0; v < V; ++v) p_t[i*V+v] = (float)(p_t[i*V+v] / St);
        // KL(P||Q) = sum P*(log P - log Q)
        for (int64_t v = 0; v < V; ++v) {
            if (p_t[i*V+v] > 1e-20f) {
                L += p_t[i*V+v] * (logf(p_t[i*V+v]) - logf(q_s[i*V+v] + 1e-20f));
            }
        }
    }
    // Apply T^2 + mean over positions
    L = L * (T * T) / (double)BT;
    ((float*)out->data)[0] = (float)L;
    slate_tensor_t* inputs[1] = {student};
    slate_graph_node_t* node = slate_graph_record(ctx, "kd_loss", inputs, 1, out, kd_backward);
    if (node) {
        kd_state_t* st = (kd_state_t*)slate_arena_alloc(ctx->scratch_arena, sizeof(*st), 16);
        st->p_teacher = p_t; st->q_student = q_s;
        st->T = T; st->BT = BT; st->V = V;
        node->user_data = st;
    }
    if (!out->grad) out->grad = slate_arena_alloc(ctx->scratch_arena, sizeof(float), 16);
    out->requires_grad = (node != NULL) || student->requires_grad;
    return out;
}

// ============================================================================
// Top-k variant: teacher distribution is sparse (support of size K << V).
// ============================================================================

typedef struct {
    float*   q_student;     // [BT, V]   student softmax(s/T)
    float*   p_topk;        // [BT, K]   teacher softmax(topk_logits / T) over K
    int32_t* topk_indices;  // [BT, K]   vocab indices (alias into input tensor data)
    float    T;
    int64_t  BT, V, K;
} kd_topk_state_t;

static void kd_topk_backward(slate_graph_node_t* node) {
    slate_tensor_t* s = node->inputs[0];
    slate_tensor_t* out = node->output;
    if (!s->requires_grad || !s->grad) return;
    kd_topk_state_t* st = (kd_topk_state_t*)node->user_data;
    float dy = ((const float*)out->grad)[0];
    float scale = (dy / (float)st->BT) * st->T;
    float* dl = (float*)s->grad;
    // For every position, gradient is T/(BT) * (Q_v - P~_v).
    // P~ is sparse so we add Q_v everywhere, then subtract P_k at top-k slots.
    for (int64_t i = 0; i < st->BT; ++i) {
        const float* q = st->q_student + i * st->V;
        float* dr = dl + i * st->V;
        for (int64_t v = 0; v < st->V; ++v) dr[v] += scale * q[v];
        const float*   p = st->p_topk + i * st->K;
        const int32_t* ix = st->topk_indices + i * st->K;
        for (int64_t k = 0; k < st->K; ++k) {
            int32_t v = ix[k];
            if (v >= 0 && v < st->V) dr[v] -= scale * p[k];
        }
    }
}

slate_tensor_t* slate_op_kd_loss_topk(slate_graph_ctx_t* ctx,
                                       slate_tensor_t* student,
                                       slate_tensor_t* topk_indices,
                                       slate_tensor_t* topk_logits,
                                       float T) {
    if (!ctx || !student || !topk_indices || !topk_logits) return NULL;
    if (student->n_dims != 3 || topk_indices->n_dims != 3 || topk_logits->n_dims != 3) return NULL;
    if (student->dtype != SLATE_DTYPE_F32) return NULL;
    if (topk_indices->dtype != SLATE_DTYPE_I32) return NULL;
    if (topk_logits->dtype != SLATE_DTYPE_F32) return NULL;
    int64_t B = student->shape[0], Tt = student->shape[1], V = student->shape[2];
    if (topk_indices->shape[0] != B || topk_indices->shape[1] != Tt) return NULL;
    if (topk_logits->shape[0] != B || topk_logits->shape[1] != Tt) return NULL;
    if (topk_indices->shape[2] != topk_logits->shape[2]) return NULL;
    int64_t K = topk_indices->shape[2];
    int64_t BT = B * Tt;

    int64_t s1[1] = {1};
    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32, 1, s1, false);
    float* q_s = (float*)slate_arena_alloc(ctx->scratch_arena, (size_t)BT*V*sizeof(float), 16);
    float* p_k = (float*)slate_arena_alloc(ctx->scratch_arena, (size_t)BT*K*sizeof(float), 16);

    const float*   ps = (const float*)student->data;
    const int32_t* ix = (const int32_t*)topk_indices->data;
    const float*   tl = (const float*)topk_logits->data;
    double L = 0;
    for (int64_t i = 0; i < BT; ++i) {
        // softmax(student / T) over full vocab
        const float* row_s = ps + i * V;
        float m_s = row_s[0] / T;
        for (int64_t v = 1; v < V; ++v) { float x = row_s[v]/T; if (x > m_s) m_s = x; }
        double Ss = 0;
        float* q_row = q_s + i * V;
        for (int64_t v = 0; v < V; ++v) { q_row[v] = expf(row_s[v]/T - m_s); Ss += q_row[v]; }
        float invS = (float)(1.0 / Ss);
        for (int64_t v = 0; v < V; ++v) q_row[v] *= invS;

        // softmax(topk_logits / T) — over K, gives P~ on the support
        const float*   tl_row = tl + i * K;
        const int32_t* ix_row = ix + i * K;
        float m_t = tl_row[0] / T;
        for (int64_t k = 1; k < K; ++k) { float x = tl_row[k]/T; if (x > m_t) m_t = x; }
        double St = 0;
        float* p_row = p_k + i * K;
        for (int64_t k = 0; k < K; ++k) { p_row[k] = expf(tl_row[k]/T - m_t); St += p_row[k]; }
        float invSt = (float)(1.0 / St);
        for (int64_t k = 0; k < K; ++k) p_row[k] *= invSt;

        // KL contribution: sum_k P_k * (log P_k - log Q[ix_k])
        for (int64_t k = 0; k < K; ++k) {
            if (p_row[k] < 1e-20f) continue;
            int32_t v = ix_row[k];
            if (v < 0 || v >= V) continue;   // skip invalid indices (e.g. -1 padding)
            float q_v = q_row[v];
            L += (double)p_row[k] * (logf(p_row[k]) - logf(q_v + 1e-20f));
        }
    }
    L = L * (T * T) / (double)BT;
    ((float*)out->data)[0] = (float)L;

    slate_tensor_t* inputs[1] = { student };
    slate_graph_node_t* node = slate_graph_record(ctx, "kd_loss_topk", inputs, 1, out, kd_topk_backward);
    if (node) {
        kd_topk_state_t* st = (kd_topk_state_t*)slate_arena_alloc(ctx->scratch_arena, sizeof(*st), 16);
        st->q_student     = q_s;
        st->p_topk        = p_k;
        st->topk_indices  = (int32_t*)topk_indices->data;
        st->T  = T;
        st->BT = BT;
        st->V  = V;
        st->K  = K;
        node->user_data = st;
    }
    if (!out->grad) out->grad = slate_arena_alloc(ctx->scratch_arena, sizeof(float), 16);
    out->requires_grad = (node != NULL) || student->requires_grad;
    return out;
}
