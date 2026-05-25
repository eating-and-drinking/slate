// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// quant.h — GGML quantized format dequantizers.
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

#ifdef __cplusplus
}
#endif

#endif
