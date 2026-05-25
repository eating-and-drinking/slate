// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// add_bias.c — y[..., c] = x[..., c] + b[c].
//
// Treats x as [N, C] where N is the product of leading dims. Backward sums
// upstream gradient over N into b's grad.

#include "slate/ops.h"
#include "slate/tensor.h"
#include "slate/autograd.h"

static void add_bias_backward(slate_graph_node_t* node) {
    slate_tensor_t* x = node->inputs[0];
    slate_tensor_t* b = node->inputs[1];
    slate_tensor_t* out = node->output;
    int64_t C = b->shape[0];
    int64_t N = slate_tensor_numel(out) / C;
    const float* d_out = (const float*)out->grad;

    if (x->requires_grad && x->grad) {
        float* d_x = (float*)x->grad;
        int64_t n = slate_tensor_numel(out);
        for (int64_t i = 0; i < n; ++i) d_x[i] += d_out[i];
    }
    if (b->requires_grad && b->grad) {
        float* d_b = (float*)b->grad;
        for (int64_t i = 0; i < N; ++i) {
            const float* row = d_out + i * C;
            for (int64_t c = 0; c < C; ++c) d_b[c] += row[c];
        }
    }
}

slate_tensor_t* slate_op_add_bias(slate_graph_ctx_t* ctx,
                                  slate_tensor_t* x,
                                  slate_tensor_t* b) {
    if (!ctx || !x || !b) return NULL;
    if (x->dtype != SLATE_DTYPE_F32 || b->dtype != SLATE_DTYPE_F32) return NULL;
    if (b->n_dims != 1) return NULL;
    if (x->shape[x->n_dims - 1] != b->shape[0]) return NULL;

    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32,
                                            x->n_dims, x->shape, false);
    if (!out) return NULL;

    int64_t C = b->shape[0];
    int64_t N = slate_tensor_numel(out) / C;
    const float* px = (const float*)x->data;
    const float* pb = (const float*)b->data;
    float* po = (float*)out->data;
    for (int64_t n = 0; n < N; ++n) {
        const float* row = px + n * C;
        float* orow = po + n * C;
        for (int64_t c = 0; c < C; ++c) orow[c] = row[c] + pb[c];
    }

    slate_tensor_t* inputs[2] = {x, b};
    slate_graph_node_t* node = slate_graph_record(ctx, "add_bias", inputs, 2, out,
                                                   add_bias_backward);
    if (node && out->requires_grad && !out->grad) {
        out->grad = slate_arena_alloc(ctx->scratch_arena,
                                       (size_t)slate_tensor_numel(out) * sizeof(float),
                                       16);
    }
    return out;
}
