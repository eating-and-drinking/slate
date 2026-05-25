// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// rms_norm.c — RMSNorm with learnable scale w. Operates over the last dim.
//
//   rms = sqrt(mean_i(x[i]^2) + eps)
//   y[i] = (x[i] / rms) * w[i]
//
// AVX2 SIMD for C >= 16; bit-identical original scalar code for smaller C
// so that finite-difference gradient checks see no precision regression.

#include "slate/ops.h"
#include "slate/tensor.h"
#include "slate/autograd.h"
#include "slate/runtime.h"
#include "simd_helpers.h"

#include <math.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define SIMD_MIN_C 16

typedef struct rms_state {
    float* rms_cache;
    float eps;
    int64_t n_rows;
    int64_t C;
} rms_state_t;

static void rms_norm_backward(slate_graph_node_t* node) {
    slate_tensor_t* x = node->inputs[0];
    slate_tensor_t* w = node->inputs[1];
    slate_tensor_t* out = node->output;
    rms_state_t* st = (rms_state_t*)node->user_data;
    const float* px = (const float*)x->data;
    const float* pw = (const float*)w->data;
    const float* dy = (const float*)out->grad;
    float* dx = x->requires_grad ? (float*)x->grad : NULL;
    float* dw = w->requires_grad ? (float*)w->grad : NULL;
    int64_t C = st->C;

    for (int64_t r = 0; r < st->n_rows; ++r) {
        const float* xr  = px + r * C;
        const float* dyr = dy + r * C;
        float rms = st->rms_cache[r];
        float inv_rms = 1.0f / rms;

#if defined(__AVX2__)
        if (C >= SIMD_MIN_C) {
            // s = sum_j (xr[j] * inv_rms) * pw[j] * dyr[j]
            __m256 acc = _mm256_setzero_ps();
            __m256 inv_v = _mm256_set1_ps(inv_rms);
            int64_t j = 0;
            for (; j + 8 <= C; j += 8) {
                __m256 xv = _mm256_loadu_ps(xr  + j);
                __m256 wv = _mm256_loadu_ps(pw  + j);
                __m256 dv = _mm256_loadu_ps(dyr + j);
                __m256 t  = _mm256_mul_ps(_mm256_mul_ps(xv, inv_v), wv);
                acc = _mm256_fmadd_ps(t, dv, acc);
            }
            double s_d = (double)slate_hsum256(acc);
            for (; j < C; ++j) s_d += (double)xr[j] * (double)inv_rms * (double)pw[j] * (double)dyr[j];
            float s_over_C = (float)(s_d / (double)C);

            if (dx) {
                float* dxr = dx + r * C;
                __m256 sov = _mm256_set1_ps(s_over_C);
                int64_t i = 0;
                for (; i + 8 <= C; i += 8) {
                    __m256 wv = _mm256_loadu_ps(pw  + i);
                    __m256 dv = _mm256_loadu_ps(dyr + i);
                    __m256 xv = _mm256_loadu_ps(xr  + i);
                    __m256 xhat = _mm256_mul_ps(xv, inv_v);
                    __m256 inner = _mm256_fmsub_ps(wv, dv, _mm256_mul_ps(xhat, sov));
                    __m256 add = _mm256_mul_ps(inner, inv_v);
                    __m256 prev = _mm256_loadu_ps(dxr + i);
                    _mm256_storeu_ps(dxr + i, _mm256_add_ps(prev, add));
                }
                for (; i < C; ++i) {
                    float xhat = xr[i] * inv_rms;
                    dxr[i] += (pw[i] * dyr[i] - xhat * s_over_C) * inv_rms;
                }
            }
            if (dw) {
                int64_t i = 0;
                for (; i + 8 <= C; i += 8) {
                    __m256 xv = _mm256_loadu_ps(xr  + i);
                    __m256 dv = _mm256_loadu_ps(dyr + i);
                    __m256 t  = _mm256_mul_ps(_mm256_mul_ps(xv, inv_v), dv);
                    __m256 prev = _mm256_loadu_ps(dw + i);
                    _mm256_storeu_ps(dw + i, _mm256_add_ps(prev, t));
                }
                for (; i < C; ++i) dw[i] += (xr[i] * inv_rms) * dyr[i];
            }
            continue;
        }
#endif
        // Original scalar path — bit-identical to pre-SIMD code.
        double s = 0.0;
        for (int64_t j = 0; j < C; ++j) {
            s += (double)(xr[j] * inv_rms) * (double)pw[j] * (double)dyr[j];
        }
        float s_over_C = (float)(s / (double)C);
        if (dx) {
            float* dxr = dx + r * C;
            for (int64_t i = 0; i < C; ++i) {
                float xhat = xr[i] * inv_rms;
                dxr[i] += (pw[i] * dyr[i] - xhat * s_over_C) * inv_rms;
            }
        }
        if (dw) {
            for (int64_t i = 0; i < C; ++i) {
                dw[i] += (xr[i] * inv_rms) * dyr[i];
            }
        }
    }
}

slate_tensor_t* slate_op_rms_norm(slate_graph_ctx_t* ctx,
                                   slate_tensor_t* x,
                                   slate_tensor_t* w,
                                   float eps) {
    if (!ctx || !x || !w) return NULL;
    if (x->dtype != SLATE_DTYPE_F32 || w->dtype != SLATE_DTYPE_F32) return NULL;
    if (w->n_dims != 1) return NULL;
    int64_t C = x->shape[x->n_dims - 1];
    if (w->shape[0] != C) return NULL;

    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32,
                                            x->n_dims, x->shape, false);
    if (!out) return NULL;
    int64_t n_rows = slate_tensor_numel(x) / C;
    float* rms_cache = (float*)slate_arena_alloc(ctx->scratch_arena,
                                                  (size_t)n_rows * sizeof(float), 16);

    const float* px = (const float*)x->data;
    const float* pw = (const float*)w->data;
    float* po = (float*)out->data;

    for (int64_t r = 0; r < n_rows; ++r) {
        const float* xr = px + r * C;
#if defined(__AVX2__)
        if (C >= SIMD_MIN_C) {
            __m256 acc = _mm256_setzero_ps();
            int64_t i = 0;
            for (; i + 8 <= C; i += 8) {
                __m256 v = _mm256_loadu_ps(xr + i);
                acc = _mm256_fmadd_ps(v, v, acc);
            }
            double sumsq_d = (double)slate_hsum256(acc);
            for (; i < C; ++i) sumsq_d += (double)xr[i] * (double)xr[i];
            float rms = sqrtf((float)(sumsq_d / (double)C) + eps);
            rms_cache[r] = rms;
            float inv = 1.0f / rms;
            float* orow = po + r * C;
            __m256 inv_v = _mm256_set1_ps(inv);
            int64_t j = 0;
            for (; j + 8 <= C; j += 8) {
                __m256 xv = _mm256_loadu_ps(xr + j);
                __m256 wv = _mm256_loadu_ps(pw + j);
                __m256 t  = _mm256_mul_ps(_mm256_mul_ps(xv, inv_v), wv);
                _mm256_storeu_ps(orow + j, t);
            }
            for (; j < C; ++j) orow[j] = (xr[j] * inv) * pw[j];
            continue;
        }
#endif
        // Original scalar path.
        double sumsq = 0.0;
        for (int64_t i = 0; i < C; ++i) sumsq += (double)xr[i] * (double)xr[i];
        float rms = sqrtf((float)(sumsq / (double)C) + eps);
        rms_cache[r] = rms;
        float inv = 1.0f / rms;
        float* orow = po + r * C;
        for (int64_t i = 0; i < C; ++i) orow[i] = (xr[i] * inv) * pw[i];
    }

    slate_tensor_t* inputs[2] = {x, w};
    slate_graph_node_t* node = slate_graph_record(ctx, "rms_norm", inputs, 2,
                                                   out, rms_norm_backward);
    if (node) {
        rms_state_t* st = (rms_state_t*)slate_arena_alloc(ctx->scratch_arena,
                                                          sizeof(*st), 16);
        st->rms_cache = rms_cache;
        st->eps = eps;
        st->n_rows = n_rows;
        st->C = C;
        node->user_data = st;
        if (!out->grad) {
            out->grad = slate_arena_alloc(ctx->scratch_arena,
                                           (size_t)slate_tensor_numel(out) * sizeof(float),
                                           16);
        }
    }
    return out;
}
