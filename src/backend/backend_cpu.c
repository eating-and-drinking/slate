// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// backend_cpu.c — CPU implementation of slate_backend_t.
// Wraps the existing AVX2 primitives so the inference engine can be
// retargeted to GPU later without touching the engine's structure.

#include "slate/backend.h"
#include "slate/quant.h"
#include "../ops/gemm_internal.h"
#include "../ops/simd_helpers.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

static void* cpu_alloc(size_t bytes) {
    if (bytes == 0) return NULL;
    void* p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    return p;
}
static void cpu_release(void* p) { free(p); }
static void cpu_copy(void* dst, const void* src, size_t bytes) {
    memcpy(dst, src, bytes);
}

static void cpu_matvec(float* y, const float* A, const float* x, int M, int K) {
    memset(y, 0, (size_t)M * sizeof(float));
    slate_gemm_packed_accumulate(y, 1, A, K, x, 1, M, K, 1);
}

static void cpu_linear_batch(float* C, const float* X, const float* W,
                              int B, int K, int N) {
    memset(C, 0, (size_t)B * (size_t)N * sizeof(float));
    slate_gemm_packed_accumulate(C, N, X, K, W, N, B, K, N);
}

static void cpu_matvec_q4k(float* y, const void* W_q4k, const float* x, int M, int K) {
    slate_q4_k_matvec(y, W_q4k, x, M, K);
}

static void cpu_rmsnorm_row(float* y, const float* x, const float* w,
                             int D, float eps) {
    double sq = 0;
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    int d0 = 0;
    for (; d0 + 8 <= D; d0 += 8) {
        __m256 xv = _mm256_loadu_ps(x + d0);
        acc = _mm256_fmadd_ps(xv, xv, acc);
    }
    sq = (double)slate_hsum256(acc);
    for (; d0 < D; ++d0) sq += (double)x[d0] * (double)x[d0];
#else
    for (int d = 0; d < D; ++d) sq += (double)x[d] * (double)x[d];
#endif
    float inv = 1.0f / sqrtf((float)(sq / (double)D) + eps);
#if defined(__AVX2__)
    __m256 iv = _mm256_set1_ps(inv);
    int d;
    for (d = 0; d + 8 <= D; d += 8) {
        __m256 xv = _mm256_loadu_ps(x + d);
        __m256 wv = _mm256_loadu_ps(w + d);
        _mm256_storeu_ps(y + d, _mm256_mul_ps(_mm256_mul_ps(xv, wv), iv));
    }
    for (; d < D; ++d) y[d] = x[d] * w[d] * inv;
#else
    for (int d = 0; d < D; ++d) y[d] = x[d] * w[d] * inv;
#endif
}

static void cpu_silu_mul(float* y, const float* gate, const float* up, int H) {
#if defined(__AVX2__)
    int d = 0;
    for (; d + 8 <= H; d += 8) {
        __m256 g = _mm256_loadu_ps(gate + d);
        __m256 u = _mm256_loadu_ps(up   + d);
        __m256 s = slate_sigmoid256_ps(g);
        _mm256_storeu_ps(y + d, _mm256_mul_ps(_mm256_mul_ps(g, s), u));
    }
    for (; d < H; ++d) {
        float g = gate[d];
        y[d] = (g / (1.0f + expf(-g))) * up[d];
    }
#else
    for (int d = 0; d < H; ++d) {
        float g = gate[d];
        y[d] = (g / (1.0f + expf(-g))) * up[d];
    }
#endif
}

static void cpu_add_inplace(float* y, const float* x, int D) {
#if defined(__AVX2__)
    int d = 0;
    for (; d + 8 <= D; d += 8) {
        __m256 a = _mm256_loadu_ps(y + d);
        __m256 b = _mm256_loadu_ps(x + d);
        _mm256_storeu_ps(y + d, _mm256_add_ps(a, b));
    }
    for (; d < D; ++d) y[d] += x[d];
#else
    for (int d = 0; d < D; ++d) y[d] += x[d];
#endif
}

static void cpu_embed_lookup(float* y, const float* W, int token_id, int D) {
    memcpy(y, W + (int64_t)token_id * D, (size_t)D * sizeof(float));
}

static void cpu_attention_step(float* out, const float* q,
                                const float* K_cache, const float* V_cache,
                                float* scores, int L, int D) {
    float scale = 1.0f / sqrtf((float)D);
#if defined(__AVX2__)
    for (int t = 0; t < L; ++t) {
        const float* k_row = K_cache + (int64_t)t * D;
        __m256 acc = _mm256_setzero_ps();
        int d = 0;
        for (; d + 8 <= D; d += 8) {
            __m256 qv = _mm256_loadu_ps(q + d);
            __m256 kv = _mm256_loadu_ps(k_row + d);
            acc = _mm256_fmadd_ps(qv, kv, acc);
        }
        float s = slate_hsum256(acc);
        for (; d < D; ++d) s += q[d] * k_row[d];
        scores[t] = s * scale;
    }
#else
    for (int t = 0; t < L; ++t) {
        const float* k_row = K_cache + (int64_t)t * D;
        float s = 0;
        for (int d = 0; d < D; ++d) s += q[d] * k_row[d];
        scores[t] = s * scale;
    }
#endif
    float m = scores[0];
    for (int t = 1; t < L; ++t) if (scores[t] > m) m = scores[t];
    double S = 0;
    for (int t = 0; t < L; ++t) { scores[t] = expf(scores[t] - m); S += scores[t]; }
    float invS = (float)(1.0 / S);
    for (int t = 0; t < L; ++t) scores[t] *= invS;
    memset(out, 0, (size_t)D * sizeof(float));
    for (int t = 0; t < L; ++t) {
        const float* v_row = V_cache + (int64_t)t * D;
        float w = scores[t];
#if defined(__AVX2__)
        __m256 wv = _mm256_set1_ps(w);
        int d = 0;
        for (; d + 8 <= D; d += 8) {
            __m256 ov = _mm256_loadu_ps(out + d);
            __m256 vv = _mm256_loadu_ps(v_row + d);
            _mm256_storeu_ps(out + d, _mm256_fmadd_ps(wv, vv, ov));
        }
        for (; d < D; ++d) out[d] += w * v_row[d];
#else
        for (int d = 0; d < D; ++d) out[d] += w * v_row[d];
#endif
    }
}

static void cpu_sync(void) {}

static const slate_backend_t g_cpu_backend = {
    .name           = "cpu",
    .alloc          = cpu_alloc,
    .release        = cpu_release,
    .copy_h2d       = cpu_copy,
    .copy_d2h       = cpu_copy,
    .matvec         = cpu_matvec,
    .linear_batch   = cpu_linear_batch,
    .matvec_q4k     = cpu_matvec_q4k,
    .rmsnorm_row    = cpu_rmsnorm_row,
    .silu_mul       = cpu_silu_mul,
    .add_inplace    = cpu_add_inplace,
    .embed_lookup   = cpu_embed_lookup,
    .attention_step = cpu_attention_step,
    .sync           = cpu_sync,
};

const slate_backend_t* slate_backend_cpu(void) { return &g_cpu_backend; }

const slate_backend_t* slate_backend_default(void) {
    const slate_backend_t* b;
    if ((b = slate_backend_cuda())  != NULL) return b;
    if ((b = slate_backend_metal()) != NULL) return b;
    return slate_backend_cpu();
}
