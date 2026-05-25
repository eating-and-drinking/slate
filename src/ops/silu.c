// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// silu.c — SiLU (Swish) activation: y = x * sigmoid(x).
//
// Derivative:
//   dy/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
//         = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
// We save sigmoid(x) for backward in a scratch buffer.

#include "slate/ops.h"
#include "slate/tensor.h"
#include "slate/autograd.h"
#include "slate/runtime.h"
#include "simd_helpers.h"

#include <math.h>

typedef struct silu_state {
    float* sig_cache;   // [numel] = sigmoid(x)
    int64_t n;
} silu_state_t;

static float sig(float v) {
    if (v >= 0.0f) { float e = expf(-v); return 1.0f / (1.0f + e); }
    float e = expf(v); return e / (1.0f + e);
}

static void silu_backward(slate_graph_node_t* node) {
    slate_tensor_t* x = node->inputs[0];
    slate_tensor_t* out = node->output;
    if (!x->requires_grad || !x->grad) return;
    silu_state_t* st = (silu_state_t*)node->user_data;
    const float* dy = (const float*)out->grad;
    const float* vx = (const float*)x->data;
    const float* s  = st->sig_cache;
    float* d_x = (float*)x->grad;
    int64_t n = st->n;

#if defined(__AVX2__)
    __m256 one = _mm256_set1_ps(1.0f);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 dyv = _mm256_loadu_ps(dy + i);
        __m256 xv  = _mm256_loadu_ps(vx + i);
        __m256 sv  = _mm256_loadu_ps(s  + i);
        // s * (1 + x * (1 - s))
        __m256 oneMinusS = _mm256_sub_ps(one, sv);
        __m256 inner = _mm256_fmadd_ps(xv, oneMinusS, one);
        __m256 grad  = _mm256_mul_ps(sv, inner);
        // d_x += dy * grad
        __m256 prev = _mm256_loadu_ps(d_x + i);
        _mm256_storeu_ps(d_x + i, _mm256_fmadd_ps(dyv, grad, prev));
    }
    for (; i < n; ++i) d_x[i] += dy[i] * s[i] * (1.0f + vx[i] * (1.0f - s[i]));
#else
    for (int64_t i = 0; i < n; ++i) d_x[i] += dy[i] * s[i] * (1.0f + vx[i] * (1.0f - s[i]));
#endif
}

slate_tensor_t* slate_op_silu(slate_graph_ctx_t* ctx, slate_tensor_t* x) {
    if (!ctx || !x || x->dtype != SLATE_DTYPE_F32) return NULL;
    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32,
                                            x->n_dims, x->shape, false);
    if (!out) return NULL;
    int64_t n = slate_tensor_numel(out);
    float* sig_cache = (float*)slate_arena_alloc(ctx->scratch_arena,
                                                  (size_t)n * sizeof(float), 16);
    const float* px = (const float*)x->data;
    float* po = (float*)out->data;

#if defined(__AVX2__)
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(px + i);
        __m256 sv = slate_sigmoid256_ps(v);
        _mm256_storeu_ps(sig_cache + i, sv);
        _mm256_storeu_ps(po + i, _mm256_mul_ps(v, sv));
    }
    for (; i < n; ++i) {
        float s = sig(px[i]);
        sig_cache[i] = s;
        po[i] = px[i] * s;
    }
#else
    for (int64_t i = 0; i < n; ++i) {
        float s = sig(px[i]);
        sig_cache[i] = s;
        po[i] = px[i] * s;
    }
#endif

    slate_tensor_t* inputs[1] = {x};
    slate_graph_node_t* node = slate_graph_record(ctx, "silu", inputs, 1, out,
                                                   silu_backward);
    if (node) {
        silu_state_t* st = (silu_state_t*)slate_arena_alloc(ctx->scratch_arena,
                                                            sizeof(*st), 16);
        st->sig_cache = sig_cache;
        st->n = n;
        node->user_data = st;
        if (!out->grad) {
            out->grad = slate_arena_alloc(ctx->scratch_arena,
                                           (size_t)n * sizeof(float), 16);
        }
    }
    return out;
}
