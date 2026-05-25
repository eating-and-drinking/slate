// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// precision.h — half-precision conversion helpers.
//
//   f16 = IEEE 754 binary16 (1 sign, 5 exponent, 10 mantissa)
//   bf16 = "bfloat16" = upper 16 bits of fp32 (truncated)
//
// f16 is what GGUF uses for its quantization scales. bf16 is the mixed-precision
// compute format used by modern transformers (fp32 master + bf16 compute).

#ifndef SLATE_PRECISION_H
#define SLATE_PRECISION_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

float    slate_f16_to_f32(uint16_t h);
uint16_t slate_f32_to_f16(float f);

float    slate_bf16_to_f32(uint16_t b);
uint16_t slate_f32_to_bf16(float f);

// Bulk conversions (vectorized when AVX2 available).
void slate_f16_to_f32_n(float* dst, const uint16_t* src, size_t n);
void slate_f32_to_f16_n(uint16_t* dst, const float* src, size_t n);
void slate_bf16_to_f32_n(float* dst, const uint16_t* src, size_t n);
void slate_f32_to_bf16_n(uint16_t* dst, const float* src, size_t n);

#ifdef __cplusplus
}
#endif

#endif
