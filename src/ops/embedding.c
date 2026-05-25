// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// embedding.c — out[b, t, :] = weight[indices[b, t], :]
//
// Forward gathers rows. Backward scatter-adds upstream gradient back into
// weight's gradient buffer. Indices are not differentiable.

#include "slate/ops.h"
#include "slate/tensor.h"
#include "slate/autograd.h"
#include "slate/runtime.h"

#include <string.h>

typedef struct emb_state {
    int32_t* indices_cache;  // owned copy on scratch
    int64_t  total;          // B*T
    int64_t  D;
    int64_t  V;
} emb_state_t;

static void embedding_backward(slate_graph_node_t* node) {
    slate_tensor_t* W = node->inputs[0];
    slate_tensor_t* out = node->output;
    if (!W->requires_grad || !W->grad) return;
    emb_state_t* st = (emb_state_t*)node->user_data;
    const float* dy = (const float*)out->grad;
    float* dW = (float*)W->grad;
    for (int64_t i = 0; i < st->total; ++i) {
        int32_t idx = st->indices_cache[i];
        if (idx < 0 || idx >= st->V) continue;
        const float* dy_row = dy + i * st->D;
        float* dW_row = dW + (int64_t)idx * st->D;
        for (int64_t d = 0; d < st->D; ++d) dW_row[d] += dy_row[d];
    }
}

slate_tensor_t* slate_op_embedding(slate_graph_ctx_t* ctx,
                                    slate_tensor_t* weight,
                                    slate_tensor_t* indices) {
    if (!ctx || !weight || !indices) return NULL;
    if (weight->dtype != SLATE_DTYPE_F32) return NULL;
    if (indices->dtype != SLATE_DTYPE_I32) return NULL;
    if (weight->n_dims != 2) return NULL;
    if (indices->n_dims < 1) return NULL;

    int64_t V = weight->shape[0];
    int64_t D = weight->shape[1];
    int64_t total = slate_tensor_numel(indices);

    // Output shape: indices.shape + [D]
    int64_t out_shape[SLATE_MAX_DIMS];
    if (indices->n_dims + 1 > SLATE_MAX_DIMS) return NULL;
    for (int i = 0; i < indices->n_dims; ++i) out_shape[i] = indices->shape[i];
    out_shape[indices->n_dims] = D;
    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32,
                                            indices->n_dims + 1, out_shape, false);
    if (!out) return NULL;

    const int32_t* idx = (const int32_t*)indices->data;
    const float* W = (const float*)weight->data;
    float* po = (float*)out->data;
    for (int64_t i = 0; i < total; ++i) {
        int32_t k = idx[i];
        if (k < 0 || k >= V) {
            memset(po + i * D, 0, (size_t)D * sizeof(float));
        } else {
            memcpy(po + i * D, W + (int64_t)k * D, (size_t)D * sizeof(float));
        }
    }

    slate_tensor_t* inputs[2] = {weight, indices};
    slate_graph_node_t* node = slate_graph_record(ctx, "embedding", inputs, 2,
                                                   out, embedding_backward);
    if (node) {
        emb_state_t* st = (emb_state_t*)slate_arena_alloc(ctx->scratch_arena,
                                                          sizeof(*st), 16);
        int32_t* icopy = (int32_t*)slate_arena_alloc(ctx->scratch_arena,
                                                     (size_t)total * sizeof(int32_t),
                                                     16);
        memcpy(icopy, idx, (size_t)total * sizeof(int32_t));
        st->indices_cache = icopy;
        st->total = total; st->D = D; st->V = V;
        node->user_data = st;
        if (!out->grad) {
            out->grad = slate_arena_alloc(ctx->scratch_arena,
                                           (size_t)slate_tensor_numel(out) * sizeof(float),
                                           16);
        }
    }
    return out;
}
