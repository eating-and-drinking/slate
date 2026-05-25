// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// activation.c — relu, sigmoid.

#include "slate/ops.h"
#include "slate/tensor.h"
#include "slate/autograd.h"

#include <math.h>

static void relu_backward(slate_graph_node_t* node) {
    slate_tensor_t* x = node->inputs[0];
    slate_tensor_t* out = node->output;
    if (!x->requires_grad || !x->grad) return;
    int64_t n = slate_tensor_numel(out);
    const float* d_out = (const float*)out->grad;
    const float* vx = (const float*)x->data;
    float* d_x = (float*)x->grad;
    for (int64_t i = 0; i < n; ++i) {
        d_x[i] += vx[i] > 0.0f ? d_out[i] : 0.0f;
    }
}

slate_tensor_t* slate_op_relu(slate_graph_ctx_t* ctx, slate_tensor_t* x) {
    if (!ctx || !x || x->dtype != SLATE_DTYPE_F32) return NULL;

    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32,
                                            x->n_dims, x->shape, false);
    if (!out) return NULL;

    int64_t n = slate_tensor_numel(out);
    const float* px = (const float*)x->data;
    float* po = (float*)out->data;
    for (int64_t i = 0; i < n; ++i) po[i] = px[i] > 0.0f ? px[i] : 0.0f;

    slate_tensor_t* inputs[1] = {x};
    slate_graph_node_t* node = slate_graph_record(ctx, "relu", inputs, 1, out, relu_backward);
    if (node && out->requires_grad && !out->grad) {
        out->grad = slate_arena_alloc(ctx->scratch_arena,
                                       (size_t)n * sizeof(float), 16);
    }
    return out;
}

// ============================================================================
// sigmoid: backward uses the output (y * (1-y)), so we read out->data, not x.
// ============================================================================
static void sigmoid_backward(slate_graph_node_t* node) {
    slate_tensor_t* x = node->inputs[0];
    slate_tensor_t* out = node->output;
    if (!x->requires_grad || !x->grad) return;
    int64_t n = slate_tensor_numel(out);
    const float* d_out = (const float*)out->grad;
    const float* y = (const float*)out->data;
    float* d_x = (float*)x->grad;
    for (int64_t i = 0; i < n; ++i) {
        d_x[i] += d_out[i] * y[i] * (1.0f - y[i]);
    }
}

slate_tensor_t* slate_op_sigmoid(slate_graph_ctx_t* ctx, slate_tensor_t* x) {
    if (!ctx || !x || x->dtype != SLATE_DTYPE_F32) return NULL;

    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32,
                                            x->n_dims, x->shape, false);
    if (!out) return NULL;

    int64_t n = slate_tensor_numel(out);
    const float* px = (const float*)x->data;
    float* po = (float*)out->data;
    for (int64_t i = 0; i < n; ++i) {
        // Numerically stable sigmoid.
        float v = px[i];
        if (v >= 0.0f) {
            float e = expf(-v);
            po[i] = 1.0f / (1.0f + e);
        } else {
            float e = expf(v);
            po[i] = e / (1.0f + e);
        }
    }

    slate_tensor_t* inputs[1] = {x};
    slate_graph_node_t* node = slate_graph_record(ctx, "sigmoid", inputs, 1, out, sigmoid_backward);
    if (node && out->requires_grad && !out->grad) {
        out->grad = slate_arena_alloc(ctx->scratch_arena,
                                       (size_t)n * sizeof(float), 16);
    }
    return out;
}
