// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// bmm.c — batch matmul, 3D or 4D. Treats last two dims as matmul, leading
// dims as batch. Both operands must agree on batch shape.

#include "slate/ops.h"
#include "slate/tensor.h"
#include "slate/autograd.h"
#include "slate/runtime.h"
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

static void mm2d(float* out, const float* a, const float* b,
                  int64_t M, int64_t K, int64_t N) {
    memset(out, 0, (size_t)(M * N) * sizeof(float));
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            float aik = a[i * K + k];
            const float* brow = b + k * N;
            float* orow = out + i * N;
            int64_t j;
#if defined(__AVX2__)
            __m256 av = _mm256_set1_ps(aik);
            for (j = 0; j + 8 <= N; j += 8) {
                __m256 ov = _mm256_loadu_ps(orow + j);
                __m256 bv = _mm256_loadu_ps(brow + j);
                ov = _mm256_fmadd_ps(av, bv, ov);
                _mm256_storeu_ps(orow + j, ov);
            }
            for (; j < N; ++j) orow[j] += aik * brow[j];
#else
            for (j = 0; j < N; ++j) orow[j] += aik * brow[j];
#endif
        }
    }
}

static int64_t leading_prod(const slate_tensor_t* t) {
    int64_t p = 1;
    for (int i = 0; i < t->n_dims - 2; ++i) p *= t->shape[i];
    return p;
}

static void bmm_backward(slate_graph_node_t* node) {
    slate_tensor_t* a = node->inputs[0];
    slate_tensor_t* b = node->inputs[1];
    slate_tensor_t* out = node->output;
    int64_t M = a->shape[a->n_dims - 2];
    int64_t K = a->shape[a->n_dims - 1];
    int64_t N = b->shape[b->n_dims - 1];
    int64_t L = leading_prod(out);
    const float* dy = (const float*)out->grad;
    const float* pa = (const float*)a->data;
    const float* pb = (const float*)b->data;
    if (a->requires_grad && a->grad) {
        float* da = (float*)a->grad;
        for (int64_t s = 0; s < L; ++s) {
            const float* dy_s = dy + s * M * N;
            const float* b_s = pb + s * K * N;
            float* da_s = da + s * M * K;
            for (int64_t i = 0; i < M; ++i) {
                for (int64_t k = 0; k < K; ++k) {
                    float acc = 0;
                    const float* dy_row = dy_s + i * N;
                    const float* b_row = b_s + k * N;
                    for (int64_t j = 0; j < N; ++j) acc += dy_row[j] * b_row[j];
                    da_s[i * K + k] += acc;
                }
            }
        }
    }
    if (b->requires_grad && b->grad) {
        float* db = (float*)b->grad;
        for (int64_t s = 0; s < L; ++s) {
            const float* a_s = pa + s * M * K;
            const float* dy_s = dy + s * M * N;
            float* db_s = db + s * K * N;
            for (int64_t k = 0; k < K; ++k) {
                for (int64_t j = 0; j < N; ++j) {
                    float acc = 0;
                    for (int64_t i = 0; i < M; ++i)
                        acc += a_s[i * K + k] * dy_s[i * N + j];
                    db_s[k * N + j] += acc;
                }
            }
        }
    }
}

slate_tensor_t* slate_op_bmm(slate_graph_ctx_t* ctx,
                              slate_tensor_t* a, slate_tensor_t* b) {
    if (!ctx || !a || !b) return NULL;
    if (a->dtype != SLATE_DTYPE_F32 || b->dtype != SLATE_DTYPE_F32) return NULL;
    if (a->n_dims != b->n_dims || a->n_dims < 3) return NULL;
    if (a->shape[a->n_dims - 1] != b->shape[b->n_dims - 2]) return NULL;
    for (int i = 0; i < a->n_dims - 2; ++i)
        if (a->shape[i] != b->shape[i]) return NULL;

    int64_t M = a->shape[a->n_dims - 2];
    int64_t K = a->shape[a->n_dims - 1];
    int64_t N = b->shape[b->n_dims - 1];
    int64_t os[SLATE_MAX_DIMS];
    for (int i = 0; i < a->n_dims - 2; ++i) os[i] = a->shape[i];
    os[a->n_dims - 2] = M; os[a->n_dims - 1] = N;
    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32, a->n_dims, os, false);
    if (!out) return NULL;
    int64_t L = leading_prod(out);
    const float* pa = (const float*)a->data;
    const float* pb = (const float*)b->data;
    float* po = (float*)out->data;
    for (int64_t s = 0; s < L; ++s) {
        mm2d(po + s * M * N, pa + s * M * K, pb + s * K * N, M, K, N);
    }
    slate_tensor_t* inputs[2] = {a, b};
    slate_graph_node_t* node = slate_graph_record(ctx, "bmm", inputs, 2, out, bmm_backward);
    if (node && out->requires_grad && !out->grad) {
        out->grad = slate_arena_alloc(ctx->scratch_arena,
                                       (size_t)slate_tensor_numel(out) * sizeof(float), 16);
    }
    return out;
}
