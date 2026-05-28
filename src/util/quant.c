// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// quant.c — GGML Q8_0 / Q4_0 dequantization.

#include "slate/quant.h"
#include "slate/precision.h"
#include "../ops/simd_helpers.h"
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

// =============================================================================
// Q4_K_M (GGML 4.5-bits/weight super-block, QK_K = 256)
// =============================================================================
//
// Block layout (144 bytes per 256 weights):
//   bytes 0..1   : f16 d        (super-block scale for scales)
//   bytes 2..3   : f16 dmin     (super-block scale for mins)
//   bytes 4..15  : 12 bytes packed (scale, min) for the 8 sub-blocks of 32
//   bytes 16..143: 128 bytes of 4-bit weights (low nibble first 32, then high)
//
// GGML's standard packing for the 12 scales[] bytes — for sub-block j in [0,8):
//   if j < 4:
//     scale_j = scales[j]   & 0x3F
//     min_j   = scales[j+4] & 0x3F
//   else:
//     scale_j = (scales[j+4] & 0x0F)        | ((scales[j-4] >> 6) << 4)
//     min_j   = (scales[j+4] >> 4)          | ((scales[j  ] >> 6) << 4)
//
// Weight reconstruction:
//   x_i = d * scale_j * q_i - dmin * min_j
// where q_i is the unsigned 4-bit nibble (0..15) at qs[]'s appropriate slot.

static inline void q4k_get_scale_min(const uint8_t* s, int j,
                                       uint8_t* out_scale, uint8_t* out_min) {
    if (j < 4) {
        *out_scale = s[j]   & 0x3F;
        *out_min   = s[j+4] & 0x3F;
    } else {
        *out_scale = (s[j+4] & 0x0F) | (((s[j-4] >> 6) & 0x03) << 4);
        *out_min   = (s[j+4] >> 4)   | (((s[j  ] >> 6) & 0x03) << 4);
    }
}

void slate_dequant_q4_k(float* dst, const void* src, int64_t n) {
    const uint8_t* p = (const uint8_t*)src;
    int64_t blocks = n / SLATE_Q4_K_BLOCK_ELEMS;
    for (int64_t b = 0; b < blocks; ++b) {
        const uint8_t* blk = p + b * SLATE_Q4_K_BLOCK_SIZE;
        uint16_t d16, dm16;
        memcpy(&d16,  blk + 0, 2);
        memcpy(&dm16, blk + 2, 2);
        float d    = slate_f16_to_f32(d16);
        float dmin = slate_f16_to_f32(dm16);
        const uint8_t* scales = blk + 4;
        const uint8_t* qs     = blk + 16;
        float* o = dst + b * SLATE_Q4_K_BLOCK_ELEMS;

        // 8 sub-blocks of 32 weights each.  qs[] is packed so that for
        // sub-block j the 32 4-bit weights come from qs[j*16 .. j*16+15]
        // with low nibbles giving the first 16 weights of the sub-block
        // and high nibbles giving the next 16.  (This matches ggml's
        // dequantize_row_q4_K reference.)
        for (int j = 0; j < 8; ++j) {
            uint8_t sc, mn;
            q4k_get_scale_min(scales, j, &sc, &mn);
            float sub_d = d * (float)sc;
            float sub_m = dmin * (float)mn;
            const uint8_t* qj = qs + j * 16;
            float* oj = o + j * 32;
            for (int i = 0; i < 16; ++i) {
                int lo = qj[i] & 0x0F;
                int hi = qj[i] >> 4;
                oj[i]      = sub_d * (float)lo - sub_m;
                oj[i + 16] = sub_d * (float)hi - sub_m;
            }
        }
    }
}

// Inner product: returns sum_i Q4_K(a)[i] * y[i].
// Implementation uses the closed-form sub-block factorisation:
//   per sub-block j:  qy_j = sum_{i in j} q_i * y_i
//                     sy_j = sum_{i in j} y_i
//   per super-block:  d * sum_j (scale_j * qy_j) - dmin * sum_j (min_j * sy_j)
// AVX2 path: load 16 nibble-bytes -> unpack to 32 int32 -> convert -> FMA with y.
float slate_dot_q4_k_f32(const void* q4k_blocks, const float* y, int64_t n) {
    const uint8_t* p = (const uint8_t*)q4k_blocks;
    int64_t blocks = n / SLATE_Q4_K_BLOCK_ELEMS;
    double total = 0;

    for (int64_t b = 0; b < blocks; ++b) {
        const uint8_t* blk = p + b * SLATE_Q4_K_BLOCK_SIZE;
        uint16_t d16, dm16;
        memcpy(&d16,  blk + 0, 2);
        memcpy(&dm16, blk + 2, 2);
        float d    = slate_f16_to_f32(d16);
        float dmin = slate_f16_to_f32(dm16);
        const uint8_t* scales = blk + 4;
        const uint8_t* qs     = blk + 16;
        const float*   yb     = y + b * SLATE_Q4_K_BLOCK_ELEMS;

#if defined(__AVX2__)
        // Defer the horizontal reduction to once per super-block. Per-sub-block,
        // we keep the partial sums in __m256 form, broadcast-multiply by the
        // sub-block scale_j (or min_j), and accumulate into a super-block-wide
        // __m256. This cuts 16 hsum256 calls per super-block down to 2.
        __m256 d_acc = _mm256_setzero_ps();   // accumulates Σ scale_j · <q,y>
        __m256 m_acc = _mm256_setzero_ps();   // accumulates Σ min_j   · Σy
        const __m128i nibble_mask = _mm_set1_epi8(0x0F);

        for (int j = 0; j < 8; ++j) {
            uint8_t sc, mn;
            q4k_get_scale_min(scales, j, &sc, &mn);
            const uint8_t* qj = qs + j * 16;
            const float*   yj = yb + j * 32;

            __m128i bytes = _mm_loadu_si128((const __m128i*)qj);
            __m128i lo8   = _mm_and_si128(bytes, nibble_mask);
            __m128i hi8   = _mm_and_si128(_mm_srli_epi16(bytes, 4), nibble_mask);
            // [16 int8] -> [16 int16] -> two [8 int32] -> two [8 f32]
            __m256i lo16 = _mm256_cvtepi8_epi16(lo8);
            __m256i hi16 = _mm256_cvtepi8_epi16(hi8);
            __m256 q_lo_lo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(lo16)));
            __m256 q_lo_hi = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(lo16, 1)));
            __m256 q_hi_lo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(hi16)));
            __m256 q_hi_hi = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(hi16, 1)));

            __m256 y0 = _mm256_loadu_ps(yj + 0);
            __m256 y1 = _mm256_loadu_ps(yj + 8);
            __m256 y2 = _mm256_loadu_ps(yj + 16);
            __m256 y3 = _mm256_loadu_ps(yj + 24);

            // qy_partial = q_lo_lo·y0 + q_lo_hi·y1 + q_hi_lo·y2 + q_hi_hi·y3
            // (4 independent FMA chains for ILP, then SIMD lane-wise reduce.)
            __m256 qy01 = _mm256_fmadd_ps(q_lo_lo, y0,
                              _mm256_mul_ps(q_lo_hi, y1));
            __m256 qy23 = _mm256_fmadd_ps(q_hi_lo, y2,
                              _mm256_mul_ps(q_hi_hi, y3));
            __m256 qy_p = _mm256_add_ps(qy01, qy23);

            // sy_partial = y0 + y1 + y2 + y3
            __m256 sy01 = _mm256_add_ps(y0, y1);
            __m256 sy23 = _mm256_add_ps(y2, y3);
            __m256 sy_p = _mm256_add_ps(sy01, sy23);

            // Broadcast scalar scale_j / min_j and accumulate into super-block.
            __m256 sc_v = _mm256_set1_ps((float)sc);
            __m256 mn_v = _mm256_set1_ps((float)mn);
            d_acc = _mm256_fmadd_ps(sc_v, qy_p, d_acc);
            m_acc = _mm256_fmadd_ps(mn_v, sy_p, m_acc);
        }
        // Single hsum per accumulator at the super-block boundary.
        float super_d_term = slate_hsum256(d_acc);
        float super_m_term = slate_hsum256(m_acc);
        total += (double)d * (double)super_d_term
               - (double)dmin * (double)super_m_term;
#else
        double super_d_term = 0;
        double super_m_term = 0;
        for (int j = 0; j < 8; ++j) {
            uint8_t sc, mn;
            q4k_get_scale_min(scales, j, &sc, &mn);
            const uint8_t* qj = qs + j * 16;
            const float* yj = yb + j * 32;
            double qy = 0, sy = 0;
            for (int i = 0; i < 16; ++i) {
                int lo = qj[i] & 0x0F;
                int hi = qj[i] >> 4;
                qy += (double)lo * (double)yj[i]
                    + (double)hi * (double)yj[i + 16];
                sy += (double)yj[i] + (double)yj[i + 16];
            }
            super_d_term += (double)sc * qy;
            super_m_term += (double)mn * sy;
        }
        total += d * super_d_term - dmin * super_m_term;
#endif
    }
    return (float)total;
}

void slate_q4_k_matvec(float* y, const void* a_q4k, const float* x, int M, int K) {
    int64_t blocks_per_row = K / SLATE_Q4_K_BLOCK_ELEMS;
    const uint8_t* base = (const uint8_t*)a_q4k;
    for (int m = 0; m < M; ++m) {
        const void* row = base + (size_t)m * (size_t)blocks_per_row * SLATE_Q4_K_BLOCK_SIZE;
        y[m] = slate_dot_q4_k_f32(row, x, K);
    }
}
