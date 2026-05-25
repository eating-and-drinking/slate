// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// gemm_internal.h — private kernel surface shared between matmul.c and bmm.c.
//
// This is NOT a public API. Lives in src/ops/ rather than include/slate/.
// It exists so the 2D packed-panel kernel from matmul.c can be reused by
// batched-matmul (bmm) without duplicating ~250 lines of code.
//
// Semantics:
//
//   slate_gemm_packed_accumulate(C, ldc, A, lda, B, ldb, M, K, N)
//
// Computes C[0:M, 0:N] += A[0:M, 0:K] @ B[0:K, 0:N] using the cache-blocked
// packed-panel kernel with 8x8 AVX2 microkernel.  Single-threaded; caller is
// expected to parallelise at the band or batch level.
//
//   - C is row-major with row stride `ldc` (units of float).
//   - A is row-major with row stride `lda`.
//   - B is row-major with row stride `ldb`.
//   - C is NOT zeroed by this call.  Initialise C before invocation if you
//     want an unaccumulated result (memset 0 then call).
//
// The function uses per-thread TLS scratch buffers; if the OS denies the
// (one-time) aligned_alloc, the routine falls back to a correct scalar loop.

#ifndef SLATE_OPS_GEMM_INTERNAL_H
#define SLATE_OPS_GEMM_INTERNAL_H

#include <stdint.h>

void slate_gemm_packed_accumulate(
    float* C, int64_t ldc,
    const float* A, int64_t lda,
    const float* B, int64_t ldb,
    int M, int K, int N);

// Out-of-place row-major transpose. Helper for backward passes that turn
// `d_a = d_out @ b^T` into a standard GEMM with bT materialised.
//
//   src: rows x cols  ->  dst: cols x rows
void slate_transpose_f32(const float* src, float* dst, int rows, int cols);

#endif
