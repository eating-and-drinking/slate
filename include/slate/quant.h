// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// quant.h — GGML quantized format dequantizers + direct quant×f32 kernels.
//
// Q8_0 block format (34 bytes):
//   d : f16 scale
//   q : 32 int8 quantized values
//   x_i = d * q_i
//
// Q4_0 block format (18 bytes):
//   d : f16 scale
//   q : 16 nibbles (32 4-bit values, packed lo|hi per byte)
//   x_i = d * (q_i - 8)  where q_i is uint4 in [0, 15]

#ifndef SLATE_QUANT_H
#define SLATE_QUANT_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Block sizes in bytes.
#define SLATE_Q8_0_BLOCK_SIZE 34
#define SLATE_Q4_0_BLOCK_SIZE 18
#define SLATE_QUANT_BLOCK_ELEMS 32

// Dequantize `n_elements` (must be multiple of 32) from packed quant blocks
// into a contiguous f32 buffer.
void slate_dequant_q8_0(float* dst, const void* src, int64_t n_elements);
void slate_dequant_q4_0(float* dst, const void* src, int64_t n_elements);

// Direct dot product: sum_i (q8[i] * x[i]) for n_elements (multiple of 32).
// Skips the dequant-to-f32 round-trip. Uses AVX2 SIMD when available.
// `q8_blocks` points at packed 34-byte Q8_0 blocks; `x` is a contiguous f32
// vector of length `n_elements`.
float slate_dot_q8_0_f32(const void* q8_blocks, const float* x, int64_t n_elements);

// y[m] = sum_k Q8_0(a[m, k]) * x[k]   for each row m in [0, M).
// `a_q8` lays out rows contiguously: row m at offset m * (K/32) * 34 bytes.
// Equivalent to dequantising A and running a matvec against x, but with a
// single fused traversal and no f32 materialisation of A.
void slate_q8_0_matvec(float* y, const void* a_q8, const float* x, int M, int K);

#ifdef __cplusplus
}
#endif

#endif
