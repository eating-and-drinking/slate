// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// simd_neon.c — ARM NEON intrinsics for the inner matmul kernel.
// Active only when compiled with -march=armv8-a or equivalent on aarch64.
// On x86 this whole file compiles to nothing (the function is omitted).

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#include <stdint.h>
#include <string.h>

// Same signature as matmul_row_axpy in src/ops/matmul.c, NEON variant.
// out[i] += sum_k a[k] * b[k*N + i] for i in 0..N
void slate_matmul_row_axpy_neon(float* out_row, const float* a_row,
                                 const float* b, int64_t K, int64_t N) {
    for (int64_t k = 0; k < K; ++k) {
        float a_ik = a_row[k];
        float32x4_t av = vdupq_n_f32(a_ik);
        const float* brow = b + k * N;
        int64_t j;
        for (j = 0; j + 4 <= N; j += 4) {
            float32x4_t ov = vld1q_f32(out_row + j);
            float32x4_t bv = vld1q_f32(brow + j);
            ov = vfmaq_f32(ov, av, bv);     // out += a * b
            vst1q_f32(out_row + j, ov);
        }
        for (; j < N; ++j) out_row[j] += a_ik * brow[j];
    }
}
#else
// On non-ARM platforms this translation unit is empty.
// (We still ship it so the build system can include it unconditionally.)
typedef int slate_neon_placeholder_t;
#endif
