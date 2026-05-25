// SPDX-License-Identifier: Apache-2.0
// permute12.c — swap dim 1 and dim 2 of a 4D tensor.

#include "slate/ops.h"
#include "slate/tensor.h"
#include "slate/autograd.h"
#include "slate/runtime.h"
#include <string.h>

slate_tensor_t* slate_op_permute_12(slate_graph_ctx_t*, slate_tensor_t*);

// out[b, y, x, d] = in[b, x, y, d]
static void perm12_kernel(const float* src, float* dst,
                           int64_t B, int64_t X, int64_t Y, int64_t D) {
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t x = 0; x < X; ++x) {
            for (int64_t y = 0; y < Y; ++y) {
                const float* s = src + ((b * X + x) * Y + y) * D;
                float* d = dst + ((b * Y + y) * X + x) * D;
                memcpy(d, s, (size_t)D * sizeof(float));
            }
        }
    }
}

static void perm12_backward(slate_graph_node_t* node) {
    slate_tensor_t* x = node->inputs[0];
    slate_tensor_t* out = node->output;
    if (!x->requires_grad || !x->grad) return;
    // grad of permute(swap 1,2) is permute(swap 1,2) again.
    int64_t B = out->shape[0], Y = out->shape[1], X = out->shape[2], D = out->shape[3];
    perm12_kernel((const float*)out->grad, (float*)x->grad, B, Y, X, D);
}

slate_tensor_t* slate_op_permute_12(slate_graph_ctx_t* ctx, slate_tensor_t* x) {
    if (!ctx || !x || x->dtype != SLATE_DTYPE_F32) return NULL;
    if (x->n_dims != 4) return NULL;
    int64_t B = x->shape[0], X = x->shape[1], Y = x->shape[2], D = x->shape[3];
    int64_t os[4] = {B, Y, X, D};
    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32, 4, os, false);
    if (!out) return NULL;
    perm12_kernel((const float*)x->data, (float*)out->data, B, X, Y, D);
    slate_tensor_t* inputs[1] = {x};
    slate_graph_node_t* node = slate_graph_record(ctx, "permute_12", inputs, 1, out, perm12_backward);
    if (node && out->requires_grad && !out->grad) {
        out->grad = slate_arena_alloc(ctx->scratch_arena,
                                       (size_t)slate_tensor_numel(out) * sizeof(float), 16);
    }
    return out;
}
