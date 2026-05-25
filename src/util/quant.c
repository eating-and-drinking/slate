// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// quant.c — GGML Q8_0 / Q4_0 dequantization.

#include "slate/quant.h"
#include "slate/precision.h"
#include <stdint.h>
#include <string.h>

void slate_dequant_q8_0(float* dst, const void* src, int64_t n) {
    const uint8_t* p = (const uint8_t*)src;
    int64_t blocks = n / SLATE_QUANT_BLOCK_ELEMS;
    for (int64_t b = 0; b < blocks; ++b) {
        const uint8_t* blk = p + b * SLATE_Q8_0_BLOCK_SIZE;
        uint16_t scale16; memcpy(&scale16, blk, 2);
        float d = slate_f16_to_f32(scale16);
        const int8_t* q = (const int8_t*)(blk + 2);
        float* o = dst + b * SLATE_QUANT_BLOCK_ELEMS;
        for (int i = 0; i < SLATE_QUANT_BLOCK_ELEMS; ++i) {
            o[i] = d * (float)q[i];
        }
    }
}

void slate_dequant_q4_0(float* dst, const void* src, int64_t n) {
    const uint8_t* p = (const uint8_t*)src;
    int64_t blocks = n / SLATE_QUANT_BLOCK_ELEMS;
    for (int64_t b = 0; b < blocks; ++b) {
        const uint8_t* blk = p + b * SLATE_Q4_0_BLOCK_SIZE;
        uint16_t scale16; memcpy(&scale16, blk, 2);
        float d = slate_f16_to_f32(scale16);
        const uint8_t* q = blk + 2;
        float* o = dst + b * SLATE_QUANT_BLOCK_ELEMS;
        // 16 bytes encode 32 4-bit values. ggml convention: low nibbles are
        // the first 16 elements, high nibbles are the last 16 elements.
        for (int i = 0; i < 16; ++i) {
            uint8_t v = q[i];
            int lo = (int)(v & 0x0f) - 8;
            int hi = (int)(v >> 4)   - 8;
            o[i]      = d * (float)lo;
            o[i + 16] = d * (float)hi;
        }
    }
}

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// -----------------------------------------------------------------------------
// Direct Q8_0 x f32 dot product. Each Q8_0 block holds 32 int8 + 1 fp16 scale.
//
// AVX2 strategy per block of 32 elements:
//   - Load 32 int8 weights as one __m256i (32 bytes)
//   - Split into low/high halves: 16 int8 -> 16 int16 each
//   - Then split each into 8 int32 -> convert to f32 -> 4 lanes of __m256
//   - Multiply by 4 lanes of __m256 from x[i:i+32]
//   - Accumulate into 4 f32 accumulators, then sum + scale at the end
//
// Scalar fallback is a straight per-element loop.
// -----------------------------------------------------------------------------
float slate_dot_q8_0_f32(const void* q8_blocks, const float* x, int64_t n_elements) {
    const uint8_t* p = (const uint8_t*)q8_blocks;
    int64_t n_blocks = n_elements / SLATE_QUANT_BLOCK_ELEMS;
    float total = 0.0f;

#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (int64_t b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = p + b * SLATE_Q8_0_BLOCK_SIZE;
        uint16_t scale16; memcpy(&scale16, blk, 2);
        float d = slate_f16_to_f32(scale16);
        __m256 dv = _mm256_set1_ps(d);

        // Load 32 int8 quantised weights.
        __m256i qi8 = _mm256_loadu_si256((const __m256i*)(blk + 2));

        // Split low/high 16-byte halves and sign-extend to int16.
        __m128i qi8_lo = _mm256_castsi256_si128(qi8);
        __m128i qi8_hi = _mm256_extracti128_si256(qi8, 1);
        __m256i qi16_lo = _mm256_cvtepi8_epi16(qi8_lo);   // 16 x int16
        __m256i qi16_hi = _mm256_cvtepi8_epi16(qi8_hi);

        // Each int16 lane -> int32 -> f32, scaled by d.
        __m128i lo_lo = _mm256_castsi256_si128(qi16_lo);
        __m128i lo_hi = _mm256_extracti128_si256(qi16_lo, 1);
        __m128i hi_lo = _mm256_castsi256_si128(qi16_hi);
        __m128i hi_hi = _mm256_extracti128_si256(qi16_hi, 1);

        __m256 w0 = _mm256_mul_ps(dv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo_lo)));
        __m256 w1 = _mm256_mul_ps(dv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo_hi)));
        __m256 w2 = _mm256_mul_ps(dv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi_lo)));
        __m256 w3 = _mm256_mul_ps(dv, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi_hi)));

        const float* xb = x + b * SLATE_QUANT_BLOCK_ELEMS;
        __m256 x0 = _mm256_loadu_ps(xb +  0);
        __m256 x1 = _mm256_loadu_ps(xb +  8);
        __m256 x2 = _mm256_loadu_ps(xb + 16);
        __m256 x3 = _mm256_loadu_ps(xb + 24);

        acc = _mm256_fmadd_ps(w0, x0, acc);
        acc = _mm256_fmadd_ps(w1, x1, acc);
        acc = _mm256_fmadd_ps(w2, x2, acc);
        acc = _mm256_fmadd_ps(w3, x3, acc);
    }
    float tmp[8]; _mm256_storeu_ps(tmp, acc);
    for (int i = 0; i < 8; ++i) total += tmp[i];
    return total;
#else
    for (int64_t b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = p + b * SLATE_Q8_0_BLOCK_SIZE;
        uint16_t scale16; memcpy(&scale16, blk, 2);
        float d = slate_f16_to_f32(scale16);
        const int8_t* q = (const int8_t*)(blk + 2);
        const float* xb = x + b * SLATE_QUANT_BLOCK_ELEMS;
        for (int i = 0; i < SLATE_QUANT_BLOCK_ELEMS; ++i) {
            total += d * (float)q[i] * xb[i];
        }
    }
    return total;
#endif
}

void slate_q8_0_matvec(float* y, const void* a_q8, const float* x, int M, int K) {
    int64_t blocks_per_row = K / SLATE_QUANT_BLOCK_ELEMS;
    const uint8_t* base = (const uint8_t*)a_q8;
    for (int m = 0; m < M; ++m) {
        const void* row = base + (size_t)m * (size_t)blocks_per_row * SLATE_Q8_0_BLOCK_SIZE;
        y[m] = slate_dot_q8_0_f32(row, x, K);
    }
}
