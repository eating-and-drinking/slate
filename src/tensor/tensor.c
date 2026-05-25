// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// tensor.c — Tensor type implementation.

#include "slate/tensor.h"
#include "slate/error.h"

#include <stdio.h>
#include <string.h>

static void compute_contiguous_strides(slate_dtype_t dtype,
                                       int n_dims,
                                       const int64_t* shape,
                                       int64_t* stride) {
    size_t elem = slate_dtype_size(dtype);
    int64_t s = (int64_t)elem;
    for (int i = n_dims - 1; i >= 0; --i) {
        stride[i] = s;
        s *= shape[i];
    }
}

slate_tensor_t* slate_tensor_new(slate_arena_t* arena,
                                 slate_dtype_t dtype,
                                 int n_dims,
                                 const int64_t* shape,
                                 bool requires_grad) {
    if (!arena || n_dims <= 0 || n_dims > SLATE_MAX_DIMS || !shape) return NULL;

    slate_tensor_t* t = (slate_tensor_t*)slate_arena_alloc(arena, sizeof(*t), 16);
    if (!t) return NULL;

    t->dtype = dtype;
    t->n_dims = n_dims;
    for (int i = 0; i < n_dims; ++i) t->shape[i] = shape[i];
    compute_contiguous_strides(dtype, n_dims, t->shape, t->stride);

    size_t total = (size_t)slate_tensor_numel(t) * slate_dtype_size(dtype);
    t->data = slate_arena_alloc(arena, total, 16);
    if (!t->data) return NULL;

    t->requires_grad = requires_grad;
    if (requires_grad) {
        t->grad = slate_arena_alloc(arena, total, 16);
        if (!t->grad) return NULL;
    } else {
        t->grad = NULL;
    }

    t->is_view = false;
    t->grad_fn = NULL;
    return t;
}

slate_tensor_t* slate_tensor_view(slate_arena_t* arena,
                                  slate_tensor_t* src,
                                  int n_dims,
                                  const int64_t* shape,
                                  const int64_t* stride) {
    if (!arena || !src || n_dims <= 0 || n_dims > SLATE_MAX_DIMS) return NULL;

    slate_tensor_t* t = (slate_tensor_t*)slate_arena_alloc(arena, sizeof(*t), 16);
    if (!t) return NULL;

    t->dtype = src->dtype;
    t->n_dims = n_dims;
    for (int i = 0; i < n_dims; ++i) {
        t->shape[i]  = shape[i];
        t->stride[i] = stride ? stride[i] : 0;
    }
    if (!stride) compute_contiguous_strides(src->dtype, n_dims, t->shape, t->stride);

    t->data = src->data;
    t->grad = src->grad;
    t->requires_grad = src->requires_grad;
    t->is_view = true;
    t->grad_fn = NULL;
    return t;
}

int64_t slate_tensor_numel(const slate_tensor_t* t) {
    if (!t) return 0;
    int64_t n = 1;
    for (int i = 0; i < t->n_dims; ++i) n *= t->shape[i];
    return n;
}

size_t slate_tensor_nbytes(const slate_tensor_t* t) {
    if (!t) return 0;
    return (size_t)slate_tensor_numel(t) * slate_dtype_size(t->dtype);
}

bool slate_tensor_is_contiguous(const slate_tensor_t* t) {
    if (!t) return false;
    int64_t expected = (int64_t)slate_dtype_size(t->dtype);
    for (int i = t->n_dims - 1; i >= 0; --i) {
        if (t->stride[i] != expected) return false;
        expected *= t->shape[i];
    }
    return true;
}

void slate_tensor_zero(slate_tensor_t* t) {
    if (!t || !t->data) return;
    memset(t->data, 0, slate_tensor_nbytes(t));
}

void slate_tensor_zero_grad(slate_tensor_t* t) {
    if (!t || !t->grad) return;
    memset(t->grad, 0, slate_tensor_nbytes(t));
}

slate_status_t slate_tensor_set_data(slate_tensor_t* t, const void* src, size_t src_bytes) {
    if (!t || !src) return SLATE_ERR_INVALID_ARGUMENT;
    size_t need = slate_tensor_nbytes(t);
    if (src_bytes != need)
        return slate_set_error(SLATE_ERR_SHAPE_MISMATCH,
                               "set_data: %zu bytes provided, %zu expected",
                               src_bytes, need);
    memcpy(t->data, src, need);
    return SLATE_OK;
}

slate_status_t slate_tensor_get_data(const slate_tensor_t* t, void* dst, size_t dst_bytes) {
    if (!t || !dst) return SLATE_ERR_INVALID_ARGUMENT;
    size_t need = slate_tensor_nbytes(t);
    if (dst_bytes < need)
        return slate_set_error(SLATE_ERR_INVALID_ARGUMENT,
                               "get_data: dst buffer too small (%zu < %zu)",
                               dst_bytes, need);
    memcpy(dst, t->data, need);
    return SLATE_OK;
}

float* slate_tensor_at_f32(slate_tensor_t* t,
                           int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    if (!t || t->dtype != SLATE_DTYPE_F32 || !t->data) return NULL;
    int64_t idx[4] = {i0, i1, i2, i3};
    size_t off = 0;
    for (int i = 0; i < t->n_dims; ++i) {
        if (idx[i] < 0 || idx[i] >= t->shape[i]) return NULL;
        off += (size_t)idx[i] * (size_t)t->stride[i];
    }
    return (float*)((uint8_t*)t->data + off);
}

void slate_tensor_print(const slate_tensor_t* t, const char* name) {
    if (!t) { printf("%s: NULL\n", name ? name : "?"); return; }
    printf("%s [", name ? name : "tensor");
    for (int i = 0; i < t->n_dims; ++i) {
        printf("%lld%s", (long long)t->shape[i], i + 1 < t->n_dims ? "x" : "");
    }
    printf("] %s\n", slate_dtype_name(t->dtype));
    if (t->dtype != SLATE_DTYPE_F32) return;

    int64_t n = slate_tensor_numel(t);
    int64_t to_show = n < 32 ? n : 32;
    const float* p = (const float*)t->data;
    for (int64_t i = 0; i < to_show; ++i) {
        printf("  [%2lld] %8.4f\n", (long long)i, p[i]);
    }
    if (n > to_show) printf("  ... (%lld more)\n", (long long)(n - to_show));
}
