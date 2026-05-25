// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// bmm.c — batched matmul (3D or 4D). Treats the last two dims as a matmul
// and the leading dims as a batch index. Both operands must agree on batch
// shape.
//
// Each per-batch 2D matmul is dispatched to the shared packed-panel kernel
// from `gemm_internal.h` (8x8 AVX2 microkernel + cache-blocked, single-
// threaded). We parallelise across the batch dimension instead.

#include "slate/ops.h"
#include "slate/tensor.h"
#include "slate/autograd.h"
#include "slate/runtime.h"
#include "gemm_internal.h"
#include <string.h>
#include <stdlib.h>

extern slate_threadpool_t* slate_global_pool(void);

static int64_t leading_prod(const slate_tensor_t* t) {
    int64_t p = 1;
    for (int i = 0; i < t->n_dims - 2; ++i) p *= t->shape[i];
    return p;
}

// -----------------------------------------------------------------------------
// Forward: out[s] = a[s] @ b[s]  for each batch s in [0, L).
// Parallel across batches; each batch is one packed-panel GEMM.
// -----------------------------------------------------------------------------
typedef struct bmm_fwd_task {
    float* out;
    const float* a;
    const float* b;
    int64_t L, M, K, N;
} bmm_fwd_task_t;

static void bmm_fwd_worker(int task_id, int n_tasks, void* ud) {
    bmm_fwd_task_t* t = (bmm_fwd_task_t*)ud;
    int64_t per = (t->L + n_tasks - 1) / n_tasks;
    int64_t s0 = (int64_t)task_id * per;
    int64_t s1 = s0 + per; if (s1 > t->L) s1 = t->L;
    int64_t MN = t->M * t->N, MK = t->M * t->K, KN = t->K * t->N;
    for (int64_t s = s0; s < s1; ++s) {
        float* C = t->out + s * MN;
        memset(C, 0, (size_t)MN * sizeof(float));
        slate_gemm_packed_accumulate(
            C,                       t->N,
            t->a + s * MK,           t->K,
            t->b + s * KN,           t->N,
            (int)t->M, (int)t->K, (int)t->N);
    }
}

// -----------------------------------------------------------------------------
// Backward.
//   d_a[s] = d_out[s] @ b[s]^T      shape [M,K] = [M,N] @ [N,K]
//   d_b[s] = a[s]^T @ d_out[s]      shape [K,N] = [K,M] @ [M,N]
// We materialise the transpose per batch (small, cache-friendly) and route
// the multiply through the packed kernel.
// -----------------------------------------------------------------------------
typedef struct bmm_bwd_task {
    float* da;            // may be NULL if a doesn't need grad
    float* db;            // may be NULL if b doesn't need grad
    const float* a;       // [L, M, K]
    const float* b;       // [L, K, N]
    const float* dy;      // [L, M, N]
    int64_t L, M, K, N;
} bmm_bwd_task_t;

static void bmm_bwd_worker(int task_id, int n_tasks, void* ud) {
    bmm_bwd_task_t* t = (bmm_bwd_task_t*)ud;
    int64_t per = (t->L + n_tasks - 1) / n_tasks;
    int64_t s0 = (int64_t)task_id * per;
    int64_t s1 = s0 + per; if (s1 > t->L) s1 = t->L;
    int64_t MN = t->M * t->N, MK = t->M * t->K, KN = t->K * t->N;

    // Per-task transpose scratch (max of K*N or M*K floats).
    int64_t tcap = (KN > MK ? KN : MK);
    float* trbuf = (float*)malloc((size_t)tcap * sizeof(float));
    if (!trbuf) {
        // Fall back to scalar if alloc fails (correctness over speed).
        for (int64_t s = s0; s < s1; ++s) {
            const float* a_s  = t->a  + s * MK;
            const float* b_s  = t->b  + s * KN;
            const float* dy_s = t->dy + s * MN;
            if (t->da) {
                float* da_s = t->da + s * MK;
                for (int64_t i = 0; i < t->M; ++i)
                    for (int64_t k = 0; k < t->K; ++k) {
                        float acc = 0;
                        for (int64_t j = 0; j < t->N; ++j)
                            acc += dy_s[i * t->N + j] * b_s[k * t->N + j];
                        da_s[i * t->K + k] += acc;
                    }
            }
            if (t->db) {
                float* db_s = t->db + s * KN;
                for (int64_t k = 0; k < t->K; ++k)
                    for (int64_t j = 0; j < t->N; ++j) {
                        float acc = 0;
                        for (int64_t i = 0; i < t->M; ++i)
                            acc += a_s[i * t->K + k] * dy_s[i * t->N + j];
                        db_s[k * t->N + j] += acc;
                    }
            }
        }
        return;
    }

    for (int64_t s = s0; s < s1; ++s) {
        const float* a_s  = t->a  + s * MK;
        const float* b_s  = t->b  + s * KN;
        const float* dy_s = t->dy + s * MN;

        if (t->da) {
            // d_a += d_y @ b^T   ->   [M,K] = [M,N] @ [N,K]
            slate_transpose_f32(b_s, trbuf, (int)t->K, (int)t->N);  // trbuf is bT [N,K]
            slate_gemm_packed_accumulate(
                t->da + s * MK,    t->K,
                dy_s,              t->N,
                trbuf,             t->K,
                (int)t->M, (int)t->N, (int)t->K);
        }
        if (t->db) {
            // d_b += a^T @ d_y   ->   [K,N] = [K,M] @ [M,N]
            slate_transpose_f32(a_s, trbuf, (int)t->M, (int)t->K);  // trbuf is aT [K,M]
            slate_gemm_packed_accumulate(
                t->db + s * KN,    t->N,
                trbuf,             t->M,
                dy_s,              t->N,
                (int)t->K, (int)t->M, (int)t->N);
        }
    }
    free(trbuf);
}

static void bmm_backward(slate_graph_node_t* node) {
    slate_tensor_t* a = node->inputs[0];
    slate_tensor_t* b = node->inputs[1];
    slate_tensor_t* out = node->output;
    int64_t M = a->shape[a->n_dims - 2];
    int64_t K = a->shape[a->n_dims - 1];
    int64_t N = b->shape[b->n_dims - 1];
    int64_t L = leading_prod(out);

    bmm_bwd_task_t t = {
        (a->requires_grad && a->grad) ? (float*)a->grad : NULL,
        (b->requires_grad && b->grad) ? (float*)b->grad : NULL,
        (const float*)a->data,
        (const float*)b->data,
        (const float*)out->grad,
        L, M, K, N,
    };
    if (!t.da && !t.db) return;
    slate_threadpool_t* pool = slate_global_pool();
    int nt = slate_threadpool_num_threads(pool);
    int n_tasks = (L < (int64_t)nt) ? (int)L : nt;
    if (n_tasks < 1) n_tasks = 1;
    slate_threadpool_parallel_for(pool, n_tasks, bmm_bwd_worker, &t);
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

    bmm_fwd_task_t ft = {
        (float*)out->data,
        (const float*)a->data,
        (const float*)b->data,
        L, M, K, N,
    };
    slate_threadpool_t* pool = slate_global_pool();
    int nt = slate_threadpool_num_threads(pool);
    int n_tasks = (L < (int64_t)nt) ? (int)L : nt;
    if (n_tasks < 1) n_tasks = 1;
    slate_threadpool_parallel_for(pool, n_tasks, bmm_fwd_worker, &ft);

    slate_tensor_t* inputs[2] = {a, b};
    slate_graph_node_t* node = slate_graph_record(ctx, "bmm", inputs, 2, out, bmm_backward);
    if (node && out->requires_grad && !out->grad) {
        out->grad = slate_arena_alloc(ctx->scratch_arena,
                                       (size_t)slate_tensor_numel(out) * sizeof(float), 16);
    }
    return out;
}
