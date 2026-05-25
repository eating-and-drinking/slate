// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// causal_mask.c — apply scale + lower-triangular mask to a [..., T, T] tensor.

#include "slate/ops.h"
#include "slate/tensor.h"
#include "slate/autograd.h"
#include "slate/runtime.h"

#define SLATE_NEG_INF (-1e30f)

typedef struct { float scale; int64_t L; int64_t T; } mask_state_t;

static void mask_backward(slate_graph_node_t* node) {
    slate_tensor_t* x = node->inputs[0];
    slate_tensor_t* out = node->output;
    if (!x->requires_grad || !x->grad) return;
    mask_state_t* st = (mask_state_t*)node->user_data;
    const float* dy = (const float*)out->grad;
    float* dx = (float*)x->grad;
    int64_t T = st->T;
    for (int64_t s = 0; s < st->L; ++s) {
        for (int64_t i = 0; i < T; ++i) {
            for (int64_t j = 0; j < T; ++j) {
                int64_t idx = s * T * T + i * T + j;
                if (j <= i) dx[idx] += st->scale * dy[idx];
            }
        }
    }
}

slate_tensor_t* slate_op_causal_mask(slate_graph_ctx_t* ctx,
                                      slate_tensor_t* x, float scale_factor) {
    if (!ctx || !x || x->dtype != SLATE_DTYPE_F32) return NULL;
    if (x->n_dims < 2) return NULL;
    int64_t T = x->shape[x->n_dims - 1];
    if (x->shape[x->n_dims - 2] != T) return NULL;
    int64_t L = slate_tensor_numel(x) / (T * T);
    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32,
                                            x->n_dims, x->shape, false);
    if (!out) return NULL;
    const float* px = (const float*)x->data;
    float* po = (float*)out->data;
    for (int64_t s = 0; s < L; ++s) {
        for (int64_t i = 0; i < T; ++i) {
            for (int64_t j = 0; j < T; ++j) {
                int64_t idx = s * T * T + i * T + j;
                po[idx] = (j <= i) ? (scale_factor * px[idx]) : SLATE_NEG_INF;
            }
        }
    }
    slate_tensor_t* inputs[1] = {x};
    slate_graph_node_t* node = slate_graph_record(ctx, "causal_mask", inputs, 1, out, mask_backward);
    if (node) {
        mask_state_t* st = (mask_state_t*)slate_arena_alloc(ctx->scratch_arena,
                                                            sizeof(*st), 16);
        st->scale = scale_factor; st->L = L; st->T = T;
        node->user_data = st;
        if (!out->grad) {
            out->grad = slate_arena_alloc(ctx->scratch_arena,
                                           (size_t)slate_tensor_numel(out) * sizeof(float), 16);
        }
    }
    return out;
}
