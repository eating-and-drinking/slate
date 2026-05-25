// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// transpose.c — swap the last two dimensions, with data copy.
// Backward: another transpose_last2 (the operation is self-inverse).

#include "slate/ops.h"
#include "slate/tensor.h"
#include "slate/autograd.h"
#include "slate/runtime.h"

#include <string.h>

// Copy [..., M, N] -> [..., N, M].
static void transpose_last2_kernel(const float* src, float* dst,
                                    int64_t outer, int64_t M, int64_t N) {
    for (int64_t b = 0; b < outer; ++b) {
        const float* s = src + b * M * N;
        float* d = dst + b * N * M;
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                d[j * M + i] = s[i * N + j];
            }
        }
    }
}

static void transpose_last2_backward(slate_graph_node_t* node) {
    slate_tensor_t* x = node->inputs[0];
    slate_tensor_t* out = node->output;
    if (!x->requires_grad || !x->grad) return;
    int64_t N = out->shape[out->n_dims - 2];
    int64_t M = out->shape[out->n_dims - 1];
    int64_t outer = slate_tensor_numel(out) / (M * N);
    transpose_last2_kernel((const float*)out->grad, (float*)x->grad, outer, N, M);
}

slate_tensor_t* slate_op_transpose_last2(slate_graph_ctx_t* ctx, slate_tensor_t* x) {
    if (!ctx || !x || x->dtype != SLATE_DTYPE_F32) return NULL;
    if (x->n_dims < 2) return NULL;
    int64_t out_shape[SLATE_MAX_DIMS];
    for (int i = 0; i < x->n_dims; ++i) out_shape[i] = x->shape[i];
    int64_t M = x->shape[x->n_dims - 2];
    int64_t N = x->shape[x->n_dims - 1];
    out_shape[x->n_dims - 2] = N;
    out_shape[x->n_dims - 1] = M;
    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32,
                                            x->n_dims, out_shape, false);
    if (!out) return NULL;
    int64_t outer = slate_tensor_numel(x) / (M * N);
    transpose_last2_kernel((const float*)x->data, (float*)out->data, outer, M, N);

    slate_tensor_t* inputs[1] = {x};
    slate_graph_node_t* node = slate_graph_record(ctx, "transpose_last2",
                                                   inputs, 1, out,
                                                   transpose_last2_backward);
    if (node && out->requires_grad && !out->grad) {
        out->grad = slate_arena_alloc(ctx->scratch_arena,
                                       (size_t)slate_tensor_numel(out) * sizeof(float),
                                       16);
    }
    return out;
}
