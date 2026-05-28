// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// quant.h — GGML quantized format dequantizers + direct quant×f32 kernels.

#ifndef SLATE_QUANT_H
#define SLATE_QUANT_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Block sizes in bytes.
#define SLATE_Q8_0_BLOCK_SIZE 34
#define SLATE_Q4_0_BLOCK_SIZE 18
#define SLATE_Q4_K_BLOCK_SIZE 144          // 4 (d,dmin) + 12 (scales) + 128 (qs)
#define SLATE_Q4_K_BLOCK_ELEMS 256         // super-block of 256 weights
#define SLATE_QUANT_BLOCK_ELEMS 32

// Dequantize `n_elements` (must be multiple of 32) from packed quant blocks
// into a contiguous f32 buffer.
void slate_dequant_q8_0(float* dst, const void* src, int64_t n_elements);
void slate_dequant_q4_0(float* dst, const void* src, int64_t n_elements);

// Direct Q8_0 dot product (multiple of 32 elements).  AVX2 SIMD.
float slate_dot_q8_0_f32(const void* q8_blocks, const float* x, int64_t n_elements);

// y[m] = sum_k Q8_0(a[m, k]) * x[k]   for each row m in [0, M).
void slate_q8_0_matvec(float* y, const void* a_q8, const float* x, int M, int K);

// =============================================================================
// Q4_K_M (GGML "k-quant" 4-bit super-block, 4.5 bits/weight)
// =============================================================================
//
// 144 bytes per 256-element super-block. ~2x memory savings vs Q8_0.

void slate_dequant_q4_k(float* dst, const void* src, int64_t n_elements);
float slate_dot_q4_k_f32(const void* q4k_blocks, const float* y, int64_t n_elements);
void slate_q4_k_matvec(float* y, const void* a_q4k, const float* x, int M, int K);

#ifdef __cplusplus
}
#endif

#endif
