// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// softmax.c — numerically stable softmax along the last dimension, plus its
// backward via the Jacobian-vector product trick:
//
//   d_x[i] = y[i] * (d_y[i] - sum_j(y[j] * d_y[j]))
//
// which avoids materializing the full Jacobian.

#include "slate/ops.h"
#include "slate/tensor.h"
#include "slate/autograd.h"

#include <math.h>

// Treat shape as [N, C] where C is the last dim and N is the product of the rest.
static void shape_to_NC(const slate_tensor_t* t, int64_t* N, int64_t* C) {
    *C = t->shape[t->n_dims - 1];
    *N = 1;
    for (int i = 0; i < t->n_dims - 1; ++i) *N *= t->shape[i];
}

static void softmax_backward(slate_graph_node_t* node) {
    slate_tensor_t* x = node->inputs[0];
    slate_tensor_t* out = node->output;
    if (!x->requires_grad || !x->grad) return;

    int64_t N, C;
    shape_to_NC(out, &N, &C);
    const float* y = (const float*)out->data;
    const float* d_y = (const float*)out->grad;
    float* d_x = (float*)x->grad;

    for (int64_t n = 0; n < N; ++n) {
        const float* yr = y + n * C;
        const float* dr = d_y + n * C;
        float* dxr = d_x + n * C;
        // sum_j(y[j] * d_y[j])
        double s = 0.0;
        for (int64_t c = 0; c < C; ++c) s += (double)yr[c] * (double)dr[c];
        for (int64_t c = 0; c < C; ++c) dxr[c] += yr[c] * (dr[c] - (float)s);
    }
}

slate_tensor_t* slate_op_softmax(slate_graph_ctx_t* ctx, slate_tensor_t* x) {
    if (!ctx || !x || x->dtype != SLATE_DTYPE_F32) return NULL;
    if (x->n_dims < 1) return NULL;

    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32,
                                            x->n_dims, x->shape, false);
    if (!out) return NULL;

    int64_t N, C;
    shape_to_NC(x, &N, &C);
    const float* px = (const float*)x->data;
    float* po = (float*)out->data;

    for (int64_t n = 0; n < N; ++n) {
        const float* row = px + n * C;
        float* orow = po + n * C;
        float m = row[0];
        for (int64_t c = 1; c < C; ++c) if (row[c] > m) m = row[c];
        double sum = 0.0;
        for (int64_t c = 0; c < C; ++c) {
            float e = expf(row[c] - m);
            orow[c] = e;
            sum += e;
        }
        float inv = (float)(1.0 / sum);
        for (int64_t c = 0; c < C; ++c) orow[c] *= inv;
    }

    slate_tensor_t* inputs[1] = {x};
    slate_graph_node_t* node = slate_graph_record(ctx, "softmax", inputs, 1, out,
                                                   softmax_backward);
    if (node && out->requires_grad && !out->grad) {
        int64_t n = slate_tensor_numel(out);
        out->grad = slate_arena_alloc(ctx->scratch_arena,
                                       (size_t)n * sizeof(float), 16);
    }
    return out;
}
