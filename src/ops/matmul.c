// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// matmul.c — packed-panel cache-blocked GEMM with 8x8 AVX2 microkernel.
//
// Algorithm (GotoBLAS layered blocking):
//
//     for jc in [0, N) step NC:
//       for kc in [0, K) step KC:
//         pack B[kc:kc+KC, jc:jc+NC]  -> B_pack  (NR-wide panels)
//         for ic in [0, M) step MC:                                     [threaded]
//           pack A[ic:ic+MC, kc:kc+KC] -> A_pack  (MR-wide panels)
//           for ir in [0, MC) step MR:
//             for jr in [0, NC) step NR:
//                microkernel: C[ic+ir:+MR, jc+jr:+NR] += A_pack * B_pack
//
// Each thread owns an exclusive ic-band so packs and writes are race-free.
// MR=8, NR=8 keeps the entire C tile in 8 YMM accumulators.

#include "slate/ops.h"
#include "slate/tensor.h"
#include "slate/autograd.h"
#include "slate/error.h"
#include "slate/runtime.h"
#include "gemm_internal.h"
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

extern slate_threadpool_t* slate_global_pool(void);

// 8x16 microkernel: 8 rows x 16 cols of C kept in 16 YMM accumulators
// (2 __m256 per row).  Per K step: 2 B-loads + 16 FMAs + 8 broadcasts.
// This saturates the AVX2 register file (16 ymm regs).
#define MR 8
#define NR 16
#define MC 64
#define KC 128
#define NC 512

static void pack_A(const float* A, int64_t lda, int m, int k, float* Ap) {
    int n_panels = (m + MR - 1) / MR;
    for (int p = 0; p < n_panels; ++p) {
        int i0 = p * MR;
        int rows = (i0 + MR <= m) ? MR : (m - i0);
        float* dst = Ap + (size_t)p * (size_t)MR * (size_t)k;
        for (int kk = 0; kk < k; ++kk) {
            int ii = 0;
            for (; ii < rows; ++ii) dst[kk * MR + ii] = A[(i0 + ii) * lda + kk];
            for (; ii < MR; ++ii)   dst[kk * MR + ii] = 0.0f;
        }
    }
}

static void pack_B(const float* B, int64_t ldb, int k, int n, float* Bp) {
    int n_panels = (n + NR - 1) / NR;
    for (int q = 0; q < n_panels; ++q) {
        int j0 = q * NR;
        int cols = (j0 + NR <= n) ? NR : (n - j0);
        float* dst = Bp + (size_t)q * (size_t)NR * (size_t)k;
        for (int kk = 0; kk < k; ++kk) {
            int jj = 0;
#if defined(__AVX2__)
            if (cols == NR) {
                // NR=16 -> 2 x __m256 loads/stores per row.
                __m256 v0 = _mm256_loadu_ps(B + kk * ldb + j0);
                __m256 v1 = _mm256_loadu_ps(B + kk * ldb + j0 + 8);
                _mm256_storeu_ps(dst + kk * NR,     v0);
                _mm256_storeu_ps(dst + kk * NR + 8, v1);
                continue;
            }
#endif
            for (; jj < cols; ++jj) dst[kk * NR + jj] = B[kk * ldb + (j0 + jj)];
            for (; jj < NR; ++jj)   dst[kk * NR + jj] = 0.0f;
        }
    }
}

#if defined(__AVX2__)
// 8x16 microkernel: 16 YMM accumulators (2 per row, low+high halves of NR=16).
static inline void microkernel_8x16(const float* Ap, const float* Bp,
                                     float* C, int64_t ldc, int k, int m, int n) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();
    __m256 c60 = _mm256_setzero_ps(), c61 = _mm256_setzero_ps();
    __m256 c70 = _mm256_setzero_ps(), c71 = _mm256_setzero_ps();
    for (int kk = 0; kk < k; ++kk) {
        // Two halves of one B row (NR=16 floats).
        __m256 b0 = _mm256_loadu_ps(Bp + kk * NR);
        __m256 b1 = _mm256_loadu_ps(Bp + kk * NR + 8);
        const float* a = Ap + kk * MR;
        __m256 av;
        av = _mm256_set1_ps(a[0]); c00 = _mm256_fmadd_ps(av, b0, c00); c01 = _mm256_fmadd_ps(av, b1, c01);
        av = _mm256_set1_ps(a[1]); c10 = _mm256_fmadd_ps(av, b0, c10); c11 = _mm256_fmadd_ps(av, b1, c11);
        av = _mm256_set1_ps(a[2]); c20 = _mm256_fmadd_ps(av, b0, c20); c21 = _mm256_fmadd_ps(av, b1, c21);
        av = _mm256_set1_ps(a[3]); c30 = _mm256_fmadd_ps(av, b0, c30); c31 = _mm256_fmadd_ps(av, b1, c31);
        av = _mm256_set1_ps(a[4]); c40 = _mm256_fmadd_ps(av, b0, c40); c41 = _mm256_fmadd_ps(av, b1, c41);
        av = _mm256_set1_ps(a[5]); c50 = _mm256_fmadd_ps(av, b0, c50); c51 = _mm256_fmadd_ps(av, b1, c51);
        av = _mm256_set1_ps(a[6]); c60 = _mm256_fmadd_ps(av, b0, c60); c61 = _mm256_fmadd_ps(av, b1, c61);
        av = _mm256_set1_ps(a[7]); c70 = _mm256_fmadd_ps(av, b0, c70); c71 = _mm256_fmadd_ps(av, b1, c71);
        if (kk + 2 < k) {
            _mm_prefetch((const char*)(Ap + (kk + 2) * MR), _MM_HINT_T0);
            _mm_prefetch((const char*)(Bp + (kk + 2) * NR), _MM_HINT_T0);
        }
    }
    if (m == MR && n == NR) {
        #define ROW(i, lo, hi)                                                  \
            _mm256_storeu_ps(C + (i) * ldc,     _mm256_add_ps(_mm256_loadu_ps(C + (i) * ldc),     lo)); \
            _mm256_storeu_ps(C + (i) * ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + (i) * ldc + 8), hi))
        ROW(0, c00, c01); ROW(1, c10, c11); ROW(2, c20, c21); ROW(3, c30, c31);
        ROW(4, c40, c41); ROW(5, c50, c51); ROW(6, c60, c61); ROW(7, c70, c71);
        #undef ROW
        return;
    }
    float tmp[MR][NR];
    _mm256_storeu_ps(tmp[0],     c00); _mm256_storeu_ps(tmp[0] + 8, c01);
    _mm256_storeu_ps(tmp[1],     c10); _mm256_storeu_ps(tmp[1] + 8, c11);
    _mm256_storeu_ps(tmp[2],     c20); _mm256_storeu_ps(tmp[2] + 8, c21);
    _mm256_storeu_ps(tmp[3],     c30); _mm256_storeu_ps(tmp[3] + 8, c31);
    _mm256_storeu_ps(tmp[4],     c40); _mm256_storeu_ps(tmp[4] + 8, c41);
    _mm256_storeu_ps(tmp[5],     c50); _mm256_storeu_ps(tmp[5] + 8, c51);
    _mm256_storeu_ps(tmp[6],     c60); _mm256_storeu_ps(tmp[6] + 8, c61);
    _mm256_storeu_ps(tmp[7],     c70); _mm256_storeu_ps(tmp[7] + 8, c71);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) C[i * ldc + j] += tmp[i][j];
}
#define microkernel_8x8 microkernel_8x16
#else
static inline void microkernel_8x16(const float* Ap, const float* Bp,
                                     float* C, int64_t ldc, int k, int m, int n) {
    float tmp[MR][NR] = {{0}};
    for (int kk = 0; kk < k; ++kk) {
        const float* a = Ap + kk * MR;
        const float* b = Bp + kk * NR;
        for (int i = 0; i < MR; ++i)
            for (int j = 0; j < NR; ++j) tmp[i][j] += a[i] * b[j];
    }
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) C[i * ldc + j] += tmp[i][j];
}
#define microkernel_8x8 microkernel_8x16
#endif

// Per-thread persistent packing scratch. First call on each thread allocates
// MC*KC + KC*NC floats once; subsequent calls reuse them. This kills the
// malloc overhead that dominated tiny-matrix performance with threading.
static __thread float* tls_Apack = NULL;
static __thread float* tls_Bpack = NULL;

void slate_gemm_packed_accumulate(
    float* C, int64_t ldc,
    const float* A, int64_t lda,
    const float* B, int64_t ldb,
    int M, int K, int N) {
    if (!tls_Apack) tls_Apack = (float*)aligned_alloc(64, (size_t)MC * (size_t)KC * sizeof(float));
    if (!tls_Bpack) tls_Bpack = (float*)aligned_alloc(64, (size_t)KC * (size_t)NC * sizeof(float));
    float* Apack = tls_Apack;
    float* Bpack = tls_Bpack;
    if (!Apack || !Bpack) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                float acc = 0;
                for (int k = 0; k < K; ++k) acc += A[i * lda + k] * B[k * ldb + j];
                C[i * ldc + j] += acc;
            }
        return;
    }
    for (int jc = 0; jc < N; jc += NC) {
        int nc = (jc + NC <= N) ? NC : (N - jc);
        for (int kc = 0; kc < K; kc += KC) {
            int kk = (kc + KC <= K) ? KC : (K - kc);
            pack_B(B + kc * ldb + jc, ldb, kk, nc, Bpack);
            for (int ic = 0; ic < M; ic += MC) {
                int mc = (ic + MC <= M) ? MC : (M - ic);
                pack_A(A + ic * lda + kc, lda, mc, kk, Apack);
                int n_a_panels = (mc + MR - 1) / MR;
                int n_b_panels = (nc + NR - 1) / NR;
                for (int p = 0; p < n_a_panels; ++p) {
                    int m_eff = (p * MR + MR <= mc) ? MR : (mc - p * MR);
                    const float* Ap = Apack + (size_t)p * (size_t)MR * (size_t)kk;
                    for (int q = 0; q < n_b_panels; ++q) {
                        int n_eff = (q * NR + NR <= nc) ? NR : (nc - q * NR);
                        const float* Bp = Bpack + (size_t)q * (size_t)NR * (size_t)kk;
                        float* Cblk = C + (ic + p * MR) * ldc + (jc + q * NR);
                        microkernel_8x8(Ap, Bp, Cblk, ldc, kk, m_eff, n_eff);
                    }
                }
            }
        }
    }
}

typedef struct mm_task {
    float* out;
    const float* a;
    const float* b;
    int64_t M, K, N;
} mm_task_t;

static void mm_worker(int task_id, int n_tasks, void* ud) {
    mm_task_t* t = (mm_task_t*)ud;
    int64_t Mt = t->M;
    int64_t per = (Mt + n_tasks - 1) / n_tasks;
    int64_t i0 = (int64_t)task_id * per;
    int64_t i1 = i0 + per; if (i1 > Mt) i1 = Mt;
    int m_band = (int)(i1 - i0);
    if (m_band <= 0) return;
    memset(t->out + i0 * t->N, 0, (size_t)m_band * (size_t)t->N * sizeof(float));
    slate_gemm_packed_accumulate(t->out + i0 * t->N, t->N,
              t->a   + i0 * t->K, t->K,
              t->b,               t->N,
              m_band, (int)t->K, (int)t->N);
}

static void matmul_f32_threaded(float* out, const float* a, const float* b,
                                 int64_t M, int64_t K, int64_t N) {
    mm_task_t t = {out, a, b, M, K, N};
    slate_threadpool_t* pool = slate_global_pool();
    int nt = slate_threadpool_num_threads(pool);
    int max_tasks = (int)((M + MC - 1) / MC);
    int n_tasks = nt < max_tasks ? nt : max_tasks;
    if (n_tasks < 1) n_tasks = 1;
    if ((uint64_t)M * (uint64_t)K * (uint64_t)N < 64ull * 64ull * 64ull) n_tasks = 1;
    slate_threadpool_parallel_for(pool, n_tasks, mm_worker, &t);
}

void slate_transpose_f32(const float* src, float* dst, int rows, int cols) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) dst[j * rows + i] = src[i * cols + j];
}

static void matmul_backward(slate_graph_node_t* node) {
    slate_tensor_t* a = node->inputs[0];
    slate_tensor_t* b = node->inputs[1];
    slate_tensor_t* out = node->output;
    int64_t M = a->shape[0], K = a->shape[1], N = b->shape[1];
    const float* d_out = (const float*)out->grad;

    if (a->requires_grad && a->grad) {
        // d_a = d_out @ b^T -> [M,K] = [M,N] @ [N,K]
        float* bT = (float*)malloc((size_t)N * (size_t)K * sizeof(float));
        if (bT) {
            slate_transpose_f32((const float*)b->data, bT, (int)K, (int)N);
            slate_threadpool_t* pool = slate_global_pool();
            int nt = slate_threadpool_num_threads(pool);
            int max_tasks = (int)((M + MC - 1) / MC); if (max_tasks < 1) max_tasks = 1;
            int n_tasks = nt < max_tasks ? nt : max_tasks;
            int per = ((int)M + n_tasks - 1) / n_tasks;
            for (int task = 0; task < n_tasks; ++task) {
                int i0 = task * per, i1 = i0 + per;
                if (i1 > (int)M) i1 = (int)M;
                int mb = i1 - i0;
                if (mb <= 0) continue;
                slate_gemm_packed_accumulate((float*)a->grad + (size_t)i0 * (size_t)K, K,
                          d_out          + (size_t)i0 * (size_t)N, N,
                          bT,                                       K,
                          mb, (int)N, (int)K);
            }
            free(bT);
        } else {
            for (int64_t i = 0; i < M; ++i)
                for (int64_t k = 0; k < K; ++k) {
                    float acc = 0;
                    for (int64_t j = 0; j < N; ++j)
                        acc += d_out[i * N + j] * ((float*)b->data)[k * N + j];
                    ((float*)a->grad)[i * K + k] += acc;
                }
        }
    }
    if (b->requires_grad && b->grad) {
        // d_b = a^T @ d_out -> [K,N] = [K,M] @ [M,N]
        float* aT = (float*)malloc((size_t)K * (size_t)M * sizeof(float));
        if (aT) {
            slate_transpose_f32((const float*)a->data, aT, (int)M, (int)K);
            slate_threadpool_t* pool = slate_global_pool();
            int nt = slate_threadpool_num_threads(pool);
            int max_tasks = (int)((K + MC - 1) / MC); if (max_tasks < 1) max_tasks = 1;
            int n_tasks = nt < max_tasks ? nt : max_tasks;
            int per = ((int)K + n_tasks - 1) / n_tasks;
            for (int task = 0; task < n_tasks; ++task) {
                int i0 = task * per, i1 = i0 + per;
                if (i1 > (int)K) i1 = (int)K;
                int kb = i1 - i0;
                if (kb <= 0) continue;
                slate_gemm_packed_accumulate((float*)b->grad + (size_t)i0 * (size_t)N, N,
                          aT             + (size_t)i0 * (size_t)M, M,
                          d_out,                                    N,
                          kb, (int)M, (int)N);
            }
            free(aT);
        } else {
            float* db = (float*)b->grad;
            const float* pa = (const float*)a->data;
            for (int64_t k = 0; k < K; ++k)
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
