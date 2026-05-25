// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// loss.c — mse_loss.
//
// MSE = mean((pred - target)^2). Returns a scalar.

#include "slate/ops.h"
#include "slate/tensor.h"
#include "slate/autograd.h"

static bool same_shape(const slate_tensor_t* a, const slate_tensor_t* b) {
    if (a->n_dims != b->n_dims) return false;
    for (int i = 0; i < a->n_dims; ++i) if (a->shape[i] != b->shape[i]) return false;
    return true;
}

static void mse_backward(slate_graph_node_t* node) {
    slate_tensor_t* pred = node->inputs[0];
    slate_tensor_t* target = node->inputs[1];
    slate_tensor_t* out = node->output;
    if (!pred->requires_grad || !pred->grad) return;

    int64_t n = slate_tensor_numel(pred);
    const float* p = (const float*)pred->data;
    const float* t = (const float*)target->data;
    float d_out = ((const float*)out->grad)[0];
    float scale = 2.0f / (float)n * d_out;

    float* d_p = (float*)pred->grad;
    for (int64_t i = 0; i < n; ++i) {
        d_p[i] += scale * (p[i] - t[i]);
    }
}

slate_tensor_t* slate_op_mse_loss(slate_graph_ctx_t* ctx,
                                  slate_tensor_t* pred,
                                  slate_tensor_t* target) {
    if (!ctx || !pred || !target || !same_shape(pred, target)) return NULL;
    if (pred->dtype != SLATE_DTYPE_F32 || target->dtype != SLATE_DTYPE_F32) return NULL;

    int64_t scalar_shape[1] = {1};
    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32,
                                            1, scalar_shape, false);
    if (!out) return NULL;

    int64_t n = slate_tensor_numel(pred);
    const float* p = (const float*)pred->data;
    const float* t = (const float*)target->data;
    double acc = 0;
    for (int64_t i = 0; i < n; ++i) {
        double d = (double)(p[i] - t[i]);
        acc += d * d;
    }
    ((float*)out->data)[0] = (float)(acc / (double)n);

    slate_tensor_t* inputs[2] = {pred, target};
    slate_graph_node_t* node = slate_graph_record(ctx, "mse_loss", inputs, 2, out, mse_backward);

    // The loss tensor needs a grad buffer always (it's the backward seed).
    if (!out->grad) {
        out->grad = slate_arena_alloc(ctx->scratch_arena, sizeof(float), 16);
    }
    out->requires_grad = (node != NULL) || pred->requires_grad;
    return out;
}
