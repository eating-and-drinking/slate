// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// scale.c — y = c * x for a compile/runtime-fixed scalar c.  AVX2 SIMD.

#include "slate/ops.h"
#include "slate/tensor.h"
#include "slate/autograd.h"
#include "slate/runtime.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

typedef struct scale_state { float c; int64_t n; } scale_state_t;

static void scale_backward(slate_graph_node_t* node) {
    slate_tensor_t* x = node->inputs[0];
    slate_tensor_t* out = node->output;
    if (!x->requires_grad || !x->grad) return;
    scale_state_t* st = (scale_state_t*)node->user_data;
    const float* dy = (const float*)out->grad;
    float* dx = (float*)x->grad;
    int64_t n = st->n;
#if defined(__AVX2__)
    __m256 cv = _mm256_set1_ps(st->c);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 dxv = _mm256_loadu_ps(dx + i);
        __m256 dyv = _mm256_loadu_ps(dy + i);
        _mm256_storeu_ps(dx + i, _mm256_fmadd_ps(dyv, cv, dxv));
    }
    for (; i < n; ++i) dx[i] += dy[i] * st->c;
#else
    for (int64_t i = 0; i < n; ++i) dx[i] += dy[i] * st->c;
#endif
}

slate_tensor_t* slate_op_scale(slate_graph_ctx_t* ctx, slate_tensor_t* x, float c) {
    if (!ctx || !x || x->dtype != SLATE_DTYPE_F32) return NULL;
    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32,
                                            x->n_dims, x->shape, false);
    if (!out) return NULL;
    int64_t n = slate_tensor_numel(out);
    const float* px = (const float*)x->data;
    float* po = (float*)out->data;
#if defined(__AVX2__)
    __m256 cv = _mm256_set1_ps(c);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xv = _mm256_loadu_ps(px + i);
        _mm256_storeu_ps(po + i, _mm256_mul_ps(xv, cv));
    }
    for (; i < n; ++i) po[i] = c * px[i];
#else
    for (int64_t i = 0; i < n; ++i) po[i] = c * px[i];
#endif
    slate_tensor_t* inputs[1] = {x};
    slate_graph_node_t* node = slate_graph_record(ctx, "scale", inputs, 1, out,
                                                   scale_backward);
    if (node) {
        scale_state_t* st = (scale_state_t*)slate_arena_alloc(ctx->scratch_arena,
                                                              sizeof(*st), 16);
        st->c = c; st->n = n;
        node->user_data = st;
        if (!out->grad) {
            out->grad = slate_arena_alloc(ctx->scratch_arena,
                                           (size_t)n * sizeof(float), 16);
        }
    }
    return out;
}
