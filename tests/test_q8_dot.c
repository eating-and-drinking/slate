// SPDX-License-Identifier: Apache-2.0
// test_q8_dot.c — verify slate_dot_q8_0_f32 matches dequant+f32-dot bit-exactly
//                  enough for production use, and slate_q8_0_matvec matches a
//                  dequant-then-matmul reference.

#define _POSIX_C_SOURCE 200809L
#include "slate/quant.h"
#include "slate/precision.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void encode_q8_0(void* dst, const float* src, int n) {
    uint8_t* p = (uint8_t*)dst;
    int blocks = n / SLATE_QUANT_BLOCK_ELEMS;
    for (int b = 0; b < blocks; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < SLATE_QUANT_BLOCK_ELEMS; ++i) {
            float v = src[b * SLATE_QUANT_BLOCK_ELEMS + i];
            float a = v < 0 ? -v : v;
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        if (d == 0) d = 1.0f / 127.0f;
        uint16_t d16 = slate_f32_to_f16(d);
        uint8_t* blk = p + b * SLATE_Q8_0_BLOCK_SIZE;
        memcpy(blk, &d16, 2);
        int8_t* q = (int8_t*)(blk + 2);
        for (int i = 0; i < SLATE_QUANT_BLOCK_ELEMS; ++i) {
            float v = src[b * SLATE_QUANT_BLOCK_ELEMS + i] / d;
            int qi = (int)(v < 0 ? v - 0.5f : v + 0.5f);
            if (qi >  127) qi =  127;
            if (qi < -128) qi = -128;
            q[i] = (int8_t)qi;
        }
    }
}

int main(void) {
    const int K = 256;
    const int M = 8;
    srand(0);

    // Generate random f32 weights and quantise.
    float* A_f32 = (float*)malloc(M * K * sizeof(float));
    for (int i = 0; i < M * K; ++i) A_f32[i] = (float)((rand() % 2001 - 1000)) / 1000.0f;
    int blocks_per_row = K / SLATE_QUANT_BLOCK_ELEMS;
    void* A_q8 = malloc((size_t)M * blocks_per_row * SLATE_Q8_0_BLOCK_SIZE);
    for (int m = 0; m < M; ++m) {
        encode_q8_0((uint8_t*)A_q8 + (size_t)m * blocks_per_row * SLATE_Q8_0_BLOCK_SIZE,
                    A_f32 + m * K, K);
    }
    // Re-dequantise to get the *quantisation-truncated* reference.
    float* A_deq = (float*)malloc(M * K * sizeof(float));
    for (int m = 0; m < M; ++m) {
        slate_dequant_q8_0(A_deq + m * K,
                           (uint8_t*)A_q8 + (size_t)m * blocks_per_row * SLATE_Q8_0_BLOCK_SIZE,
                           K);
    }
    float* x = (float*)malloc(K * sizeof(float));
    for (int i = 0; i < K; ++i) x[i] = (float)((rand() % 2001 - 1000)) / 1000.0f;

    // Reference: y_ref[m] = sum_k A_deq[m,k] * x[k]
    float y_ref[M], y_fast[M];
    for (int m = 0; m < M; ++m) {
        double acc = 0;
        for (int k = 0; k < K; ++k) acc += A_deq[m * K + k] * x[k];
        y_ref[m] = (float)acc;
    }
    // Fast: direct q8 dot.
    slate_q8_0_matvec(y_fast, A_q8, x, M, K);

    int ok = 1;
    for (int m = 0; m < M; ++m) {
        float diff = fabsf(y_fast[m] - y_ref[m]);
        float scale = fabsf(y_ref[m]) + 1e-6f;
        float rel = diff / scale;
        printf("  row %d: ref=%9.4f fast=%9.4f rel=%.2e\n", m, y_ref[m], y_fast[m], rel);
        // 1e-4 is tight because both sides use the *same* dequantised weights
        if (rel > 1e-4f) ok = 0;
    }
    free(A_f32); free(A_q8); free(A_deq); free(x);
    printf("test_q8_dot: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
