// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// elementwise.c — add, mul.
//
// M0 simplification: both inputs must have the same shape (no broadcasting).
// M1 will add minimal trailing-dim broadcasting (bias-style).

#include "slate/ops.h"
#include "slate/tensor.h"
#include "slate/autograd.h"

#include <string.h>

static bool same_shape(const slate_tensor_t* a, const slate_tensor_t* b) {
    if (a->n_dims != b->n_dims) return false;
    for (int i = 0; i < a->n_dims; ++i) if (a->shape[i] != b->shape[i]) return false;
    return true;
}

static slate_tensor_t* alloc_grad_buffer(slate_arena_t* arena, slate_tensor_t* t) {
    if (!t->grad) {
        t->grad = slate_arena_alloc(arena,
                                    (size_t)slate_tensor_numel(t) * sizeof(float),
                                    16);
    }
    return t;
}

// ============================================================================
// add
// ============================================================================
static void add_backward(slate_graph_node_t* node) {
    slate_tensor_t* a = node->inputs[0];
    slate_tensor_t* b = node->inputs[1];
    slate_tensor_t* out = node->output;
    int64_t n = slate_tensor_numel(out);
    const float* d_out = (const float*)out->grad;

    if (a->requires_grad && a->grad) {
        float* d_a = (float*)a->grad;
        for (int64_t i = 0; i < n; ++i) d_a[i] += d_out[i];
    }
    if (b->requires_grad && b->grad) {
        float* d_b = (float*)b->grad;
        for (int64_t i = 0; i < n; ++i) d_b[i] += d_out[i];
    }
}

slate_tensor_t* slate_op_add(slate_graph_ctx_t* ctx,
                             slate_tensor_t* a,
                             slate_tensor_t* b) {
    if (!ctx || !a || !b || !same_shape(a, b)) return NULL;
    if (a->dtype != SLATE_DTYPE_F32 || b->dtype != SLATE_DTYPE_F32) return NULL;

    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32,
                                            a->n_dims, a->shape, false);
    if (!out) return NULL;

    int64_t n = slate_tensor_numel(out);
    const float* pa = (const float*)a->data;
    const float* pb = (const float*)b->data;
    float* po = (float*)out->data;
    for (int64_t i = 0; i < n; ++i) po[i] = pa[i] + pb[i];

    slate_tensor_t* inputs[2] = {a, b};
    slate_graph_node_t* node = slate_graph_record(ctx, "add", inputs, 2, out, add_backward);
    if (node && out->requires_grad) alloc_grad_buffer(ctx->scratch_arena, out);
    return out;
}

// ============================================================================
// mul
// ============================================================================
typedef struct {
    slate_tensor_t* a_saved;
    slate_tensor_t* b_saved;
} mul_state_t;

static void mul_backward(slate_graph_node_t* node) {
    slate_tensor_t* a = node->inputs[0];
    slate_tensor_t* b = node->inputs[1];
    slate_tensor_t* out = node->output;
    int64_t n = slate_tensor_numel(out);
    const float* d_out = (const float*)out->grad;
    const float* va = (const float*)a->data;
    const float* vb = (const float*)b->data;

    if (a->requires_grad && a->grad) {
        float* d_a = (float*)a->grad;
        for (int64_t i = 0; i < n; ++i) d_a[i] += d_out[i] * vb[i];
    }
    if (b->requires_grad && b->grad) {
        float* d_b = (float*)b->grad;
        for (int64_t i = 0; i < n; ++i) d_b[i] += d_out[i] * va[i];
    }
}

slate_tensor_t* slate_op_mul(slate_graph_ctx_t* ctx,
                             slate_tensor_t* a,
                             slate_tensor_t* b) {
    if (!ctx || !a || !b || !same_shape(a, b)) return NULL;
    if (a->dtype != SLATE_DTYPE_F32 || b->dtype != SLATE_DTYPE_F32) return NULL;

    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32,
                                            a->n_dims, a->shape, false);
    if (!out) return NULL;

    int64_t n = slate_tensor_numel(out);
    const float* pa = (const float*)a->data;
    const float* pb = (const float*)b->data;
    float* po = (float*)out->data;
    for (int64_t i = 0; i < n; ++i) po[i] = pa[i] * pb[i];

    slate_tensor_t* inputs[2] = {a, b};
    slate_graph_node_t* node = slate_graph_record(ctx, "mul", inputs, 2, out, mul_backward);
    if (node && out->requires_grad) alloc_grad_buffer(ctx->scratch_arena, out);
    return out;
}
