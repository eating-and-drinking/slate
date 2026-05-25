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
