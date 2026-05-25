// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// simd_helpers.h — private AVX2 intrinsic helpers shared between op
// implementations.  Internal to src/ops/; not part of the public API.

#ifndef SLATE_OPS_SIMD_HELPERS_H
#define SLATE_OPS_SIMD_HELPERS_H

#if defined(__AVX2__)
#include <immintrin.h>

// Horizontal sum of one __m256.
static inline float slate_hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// Horizontal max of one __m256.
static inline float slate_hmax256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 0x55));
    return _mm_cvtss_f32(m);
}

// Vectorised exp(x) for AVX2.  Accuracy ~1 ulp in the reduced range,
// max relative error ≈ 3e-7. Range-clamped to ±88.376 to avoid IEEE-754
// overflow in the integer exponent add.
static inline __m256 slate_exp256_ps(__m256 x) {
    const __m256 max_x = _mm256_set1_ps( 88.3762626647949f);
    const __m256 min_x = _mm256_set1_ps(-88.3762626647949f);
    x = _mm256_min_ps(x, max_x);
    x = _mm256_max_ps(x, min_x);
    const __m256 LOG2EF = _mm256_set1_ps(1.44269504088896341f);
    const __m256 LN2_HI = _mm256_set1_ps(0.693359375f);
    const __m256 LN2_LO = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 HALF   = _mm256_set1_ps(0.5f);
    __m256 fx = _mm256_fmadd_ps(x, LOG2EF, HALF);
    fx = _mm256_floor_ps(fx);
    __m256 r = _mm256_fnmadd_ps(fx, LN2_HI, x);
    r = _mm256_fnmadd_ps(fx, LN2_LO, r);
    const __m256 c1 = _mm256_set1_ps(1.9875691500e-4f);
    const __m256 c2 = _mm256_set1_ps(1.3981999507e-3f);
    const __m256 c3 = _mm256_set1_ps(8.3334519073e-3f);
    const __m256 c4 = _mm256_set1_ps(4.1665795894e-2f);
    const __m256 c5 = _mm256_set1_ps(1.6666665459e-1f);
    const __m256 c6 = _mm256_set1_ps(5.0000001201e-1f);
    __m256 y = c1;
    y = _mm256_fmadd_ps(y, r, c2);
    y = _mm256_fmadd_ps(y, r, c3);
    y = _mm256_fmadd_ps(y, r, c4);
    y = _mm256_fmadd_ps(y, r, c5);
    y = _mm256_fmadd_ps(y, r, c6);
    __m256 r2 = _mm256_mul_ps(r, r);
    y = _mm256_fmadd_ps(y, r2, r);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));
    __m256i n_i = _mm256_cvtps_epi32(fx);
    n_i = _mm256_add_epi32(n_i, _mm256_set1_epi32(127));
    n_i = _mm256_slli_epi32(n_i, 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(n_i));
}

// sigmoid(x) = 1 / (1 + exp(-x)) — uses slate_exp256_ps internally.
static inline __m256 slate_sigmoid256_ps(__m256 x) {
    __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 e = slate_exp256_ps(neg);
    __m256 d = _mm256_add_ps(_mm256_set1_ps(1.0f), e);
    return _mm256_div_ps(_mm256_set1_ps(1.0f), d);
}

#endif // __AVX2__

#endif
