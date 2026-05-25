#define _POSIX_C_SOURCE 200809L
// SPDX-License-Identifier: Apache-2.0
// bench_q8_dot.c — compare two paths for Q8_0 × f32:
//   A) dequant to f32 buffer, then f32 dot
//   B) fused slate_dot_q8_0_f32 directly on the packed blocks
// Run on a single thread to isolate kernel speed.

#include "slate/quant.h"
#include "slate/precision.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

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

static void bench(int M, int K, int iters) {
    int blocks_per_row = K / SLATE_QUANT_BLOCK_ELEMS;
    float* A = (float*)malloc((size_t)M * K * sizeof(float));
    for (int i = 0; i < M * K; ++i) A[i] = (float)((rand() % 2001 - 1000)) / 1000.0f;
    void* Aq = malloc((size_t)M * blocks_per_row * SLATE_Q8_0_BLOCK_SIZE);
    for (int m = 0; m < M; ++m) {
        encode_q8_0((uint8_t*)Aq + (size_t)m * blocks_per_row * SLATE_Q8_0_BLOCK_SIZE,
                    A + m * K, K);
    }
    float* x = (float*)malloc(K * sizeof(float));
    for (int i = 0; i < K; ++i) x[i] = (float)((rand() % 2001 - 1000)) / 1000.0f;
    float* y = (float*)malloc(M * sizeof(float));
    float* A_deq = (float*)malloc((size_t)M * K * sizeof(float));

    // Path A: dequant + f32 matvec
    double t0 = now_s();
    for (int it = 0; it < iters; ++it) {
        for (int m = 0; m < M; ++m) {
            slate_dequant_q8_0(A_deq + m * K,
                               (uint8_t*)Aq + (size_t)m * blocks_per_row * SLATE_Q8_0_BLOCK_SIZE,
                               K);
        }
        for (int m = 0; m < M; ++m) {
            float acc = 0;
            for (int k = 0; k < K; ++k) acc += A_deq[m * K + k] * x[k];
            y[m] = acc;
        }
    }
    double dt_a = now_s() - t0;

    // Path B: fused q8 dot
    t0 = now_s();
    for (int it = 0; it < iters; ++it) {
        slate_q8_0_matvec(y, Aq, x, M, K);
    }
    double dt_b = now_s() - t0;

    double flops = 2.0 * (double)M * K * iters;
    printf("  M=%4d K=%5d : dequant+f32 %.2f ms (%.2f GFLOP/s)   fused q8 %.2f ms (%.2f GFLOP/s)   speedup %.2fx\n",
           M, K,
           dt_a * 1000.0 / iters, flops / dt_a / 1e9,
           dt_b * 1000.0 / iters, flops / dt_b / 1e9,
           dt_a / dt_b);
    free(A); free(Aq); free(x); free(y); free(A_deq);
}

int main(void) {
    printf("slate Q8_0 matvec benchmark (single-thread, AVX2)\n");
    bench(  64,  256, 2000);   // small Linear
    bench( 256, 1024, 500);    // LLaMA ffn_gate proj-out at d=1024
    bench(1024, 1024, 100);    // LLaMA attn out-proj at d=1024
    bench(4096, 4096, 20);     // LLaMA-7B attn weight (d_model=4096)
    return 0;
}
