// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// matmul.c — 2D matmul with AVX2 inner kernel and threaded outer loop.
// Falls back to scalar on non-x86_64 or when -mavx2 isn't enabled.

#include "slate/ops.h"
#include "slate/tensor.h"
#include "slate/autograd.h"
#include "slate/error.h"
#include "slate/runtime.h"
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

extern slate_threadpool_t* slate_global_pool(void);

// out[i, j] += sum_k a[i, k] * b[k, j].  Single row of output, vectorized over j.
static void matmul_row_axpy(float* out_row, const float* a_row, const float* b,
                             int64_t K, int64_t N) {
    int64_t j;
#if defined(__AVX2__)
    for (int64_t k = 0; k < K; ++k) {
        float a_ik = a_row[k];
        __m256 av = _mm256_set1_ps(a_ik);
        const float* brow = b + k * N;
        for (j = 0; j + 8 <= N; j += 8) {
            __m256 ov = _mm256_loadu_ps(out_row + j);
            __m256 bv = _mm256_loadu_ps(brow + j);
            ov = _mm256_fmadd_ps(av, bv, ov);
            _mm256_storeu_ps(out_row + j, ov);
        }
        for (; j < N; ++j) out_row[j] += a_ik * brow[j];
    }
#else
    for (int64_t k = 0; k < K; ++k) {
        float a_ik = a_row[k];
        const float* brow = b + k * N;
        for (j = 0; j < N; ++j) out_row[j] += a_ik * brow[j];
    }
#endif
}

typedef struct mm_task {
    float* out;
    const float* a;
    const float* b;
    int64_t M, K, N;
} mm_task_t;

static void mm_worker(int task_id, int n_tasks, void* ud) {
    mm_task_t* t = (mm_task_t*)ud;
    // Each task processes a row range of rows.
    int64_t M = t->M;
    int64_t per = (M + n_tasks - 1) / n_tasks;
    int64_t i0 = (int64_t)task_id * per;
    int64_t i1 = i0 + per; if (i1 > M) i1 = M;
    for (int64_t i = i0; i < i1; ++i) {
        memset(t->out + i * t->N, 0, (size_t)t->N * sizeof(float));
        matmul_row_axpy(t->out + i * t->N, t->a + i * t->K, t->b, t->K, t->N);
    }
}

static void matmul_f32_threaded(float* out, const float* a, const float* b,
                                 int64_t M, int64_t K, int64_t N) {
    mm_task_t t = {out, a, b, M, K, N};
    slate_threadpool_t* pool = slate_global_pool();
    int nt = slate_threadpool_num_threads(pool);
    int n_tasks = (M < (int64_t)nt) ? (int)M : nt;
    if (n_tasks < 1) n_tasks = 1;
    slate_threadpool_parallel_for(pool, n_tasks, mm_worker, &t);
}

// d_a = d_out @ b^T, d_b = a^T @ d_out  (same threading)
static void mm_da_worker(int task_id, int n_tasks, void* ud) {
    mm_task_t* t = (mm_task_t*)ud;  // out=d_a, a=d_out [M,N], b=b [K,N]
    int64_t M = t->M, K = t->K, N = t->N;
    int64_t per = (M + n_tasks - 1) / n_tasks;
    int64_t i0 = (int64_t)task_id * per, i1 = i0 + per; if (i1 > M) i1 = M;
    for (int64_t i = i0; i < i1; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            float acc = 0;
            const float* dy_row = t->a + i * N;
            const float* b_row = t->b + k * N;
#if defined(__AVX2__)
            __m256 sum = _mm256_setzero_ps();
            int64_t j;
            for (j = 0; j + 8 <= N; j += 8) {
                __m256 dv = _mm256_loadu_ps(dy_row + j);
                __m256 bv = _mm256_loadu_ps(b_row + j);
                sum = _mm256_fmadd_ps(dv, bv, sum);
            }
            float tmp[8]; _mm256_storeu_ps(tmp, sum);
            for (int x = 0; x < 8; ++x) acc += tmp[x];
            for (; j < N; ++j) acc += dy_row[j] * b_row[j];
#else
            for (int64_t j = 0; j < N; ++j) acc += dy_row[j] * b_row[j];
#endif
            t->out[i * K + k] += acc;
        }
    }
}

static void matmul_backward(slate_graph_node_t* node) {
    slate_tensor_t* a = node->inputs[0];
    slate_tensor_t* b = node->inputs[1];
    slate_tensor_t* out = node->output;
    int64_t M = a->shape[0], K = a->shape[1], N = b->shape[1];
    const float* d_out = (const float*)out->grad;

    if (a->requires_grad && a->grad) {
        mm_task_t t = {(float*)a->grad, d_out, (const float*)b->data, M, K, N};
        slate_threadpool_parallel_for(slate_global_pool(),
            (M < 8 ? 1 : slate_threadpool_num_threads(slate_global_pool())),
            mm_da_worker, &t);
    }
    if (b->requires_grad && b->grad) {
        // d_b = a^T @ d_out :  d_b[k, j] += sum_i a[i, k] * d_out[i, j]
        // Do this as scalar/threaded over k rows.
        float* db = (float*)b->grad;
        const float* pa = (const float*)a->data;
        for (int64_t k = 0; k < K; ++k) {
            for (int64_t j = 0; j < N; ++j) {
                float acc = 0;
                for (int64_t i = 0; i < M; ++i) acc += pa[i * K + k] * d_out[i * N + j];
                db[k * N + j] += acc;
            }
        }
    }
}

slate_tensor_t* slate_op_matmul(slate_graph_ctx_t* ctx,
                                slate_tensor_t* a, slate_tensor_t* b) {
    if (!ctx || !a || !b) return NULL;
    if (a->dtype != SLATE_DTYPE_F32 || b->dtype != SLATE_DTYPE_F32) return NULL;
    if (a->n_dims != 2 || b->n_dims != 2) return NULL;
    if (a->shape[1] != b->shape[0]) return NULL;
    int64_t M = a->shape[0], K = a->shape[1], N = b->shape[1];
    int64_t os[2] = {M, N};
    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32, 2, os, false);
    if (!out) return NULL;
    matmul_f32_threaded((float*)out->data, (const float*)a->data,
                        (const float*)b->data, M, K, N);
    slate_tensor_t* inputs[2] = {a, b};
    slate_graph_node_t* node = slate_graph_record(ctx, "matmul", inputs, 2, out, matmul_backward);
    if (node && out->requires_grad && !out->grad) {
        out->grad = slate_arena_alloc(ctx->scratch_arena,
                                       (size_t)slate_tensor_numel(out) * sizeof(float), 16);
    }
    return out;
}
