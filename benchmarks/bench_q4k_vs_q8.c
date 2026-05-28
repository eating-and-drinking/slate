// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// bench_q4k_vs_q8.c — compare Q4_K_M vs Q8_0 vs f32 for a 4096×4096 matvec
// (LLaMA-7B class: an attention/FFN projection on D=4096).

#include "slate/slate.h"
#include "slate/quant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define M       4096
#define K       4096
#define REPS    10

static double now_sec(void) {
    struct timeval t; gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec / 1e6;
}

// Generate a synthetic Q4_K block (same shape as the test fixture)
static void make_q4k_row(uint8_t* dst, int n_blocks, uint64_t* rng) {
    // Just fill with deterministic-ish bytes; we don't care about exact
    // values for a perf benchmark, only the layout sizes.
    for (int b = 0; b < n_blocks; ++b) {
        uint8_t* blk = dst + b * 144;
        for (int i = 0; i < 144; ++i) {
            *rng = *rng * 6364136223846793005ULL + 1442695040888963407ULL;
            blk[i] = (uint8_t)((*rng >> 24) & 0xFF);
        }
        // Force d, dmin to small positive f16 so output is finite.
        // f16 0.01 ≈ 0x1ED1; close enough for benchmark stability.
        uint16_t d_small = 0x1C00;   // ~1.0
        memcpy(blk + 0, &d_small, 2);
        memcpy(blk + 2, &d_small, 2);
    }
}

static void make_q8_row(uint8_t* dst, int n_blocks, uint64_t* rng) {
    for (int b = 0; b < n_blocks; ++b) {
        uint8_t* blk = dst + b * 34;
        uint16_t d_small = 0x1C00;
        memcpy(blk, &d_small, 2);
        for (int i = 0; i < 32; ++i) {
            *rng = *rng * 6364136223846793005ULL + 1442695040888963407ULL;
            blk[2 + i] = (uint8_t)((*rng >> 24) & 0xFF);
        }
    }
}

int main(void) {
    printf("=== slate quant matvec: Q4_K_M vs Q8_0 vs f32 ===\n");
    printf("Shape: %d × %d, %d reps\n\n", M, K, REPS);

    uint64_t rng = 0x1234;

    // Sizes:
    int n_blocks_q4k = K / 256;
    int n_blocks_q8  = K / 32;
    size_t mem_q4k = (size_t)M * n_blocks_q4k * 144;
    size_t mem_q8  = (size_t)M * n_blocks_q8  * 34;
    size_t mem_f32 = (size_t)M * K * sizeof(float);

    printf("Memory:\n");
    printf("  f32:    %.1f MB\n", mem_f32 / 1048576.0);
    printf("  Q8_0:   %.1f MB  (%.2fx vs f32)\n",
            mem_q8 / 1048576.0, (double)mem_q8 / mem_f32);
    printf("  Q4_K_M: %.1f MB  (%.2fx vs f32, %.2fx vs Q8_0)\n\n",
            mem_q4k / 1048576.0,
            (double)mem_q4k / mem_f32,
            (double)mem_q4k / mem_q8);

    uint8_t* A_q4k = (uint8_t*)malloc(mem_q4k);
    uint8_t* A_q8  = (uint8_t*)malloc(mem_q8);
    float*   A_f32 = (float*)malloc(mem_f32);
    float*   x     = (float*)malloc((size_t)K   * sizeof(float));
    float*   y     = (float*)malloc((size_t)M   * sizeof(float));
    for (int m = 0; m < M; ++m) {
        make_q4k_row(A_q4k + (size_t)m * n_blocks_q4k * 144, n_blocks_q4k, &rng);
        make_q8_row (A_q8  + (size_t)m * n_blocks_q8  * 34,  n_blocks_q8,  &rng);
    }
    for (int k = 0; k < K; ++k) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        x[k] = ((float)((rng >> 24) & 0xFFFF) / 65536.0f - 0.5f);
    }
    // Dequant Q8 to f32 once for the f32 baseline matrix
    slate_dequant_q8_0(A_f32, A_q8, (int64_t)M * K);

    // Warm-up
    slate_q4_k_matvec(y, A_q4k, x, M, K);
    slate_q8_0_matvec(y, A_q8,  x, M, K);

    // Q4_K_M timing
    double t0 = now_sec();
    for (int r = 0; r < REPS; ++r) slate_q4_k_matvec(y, A_q4k, x, M, K);
    double t1 = now_sec();
    double q4k_ms = (t1 - t0) * 1000 / REPS;
    double q4k_gflops = 2.0 * (double)M * K / (q4k_ms * 1e-3) / 1e9;

    // Q8_0 timing
    t0 = now_sec();
    for (int r = 0; r < REPS; ++r) slate_q8_0_matvec(y, A_q8, x, M, K);
    t1 = now_sec();
    double q8_ms = (t1 - t0) * 1000 / REPS;
    double q8_gflops = 2.0 * (double)M * K / (q8_ms * 1e-3) / 1e9;

    // f32 baseline (naive matvec)
    t0 = now_sec();
    for (int r = 0; r < REPS; ++r) {
        for (int m = 0; m < M; ++m) {
            float s = 0;
            const float* row = A_f32 + (size_t)m * K;
            for (int k = 0; k < K; ++k) s += row[k] * x[k];
            y[m] = s;
        }
    }
    t1 = now_sec();
    double f32_ms = (t1 - t0) * 1000 / REPS;
    double f32_gflops = 2.0 * (double)M * K / (f32_ms * 1e-3) / 1e9;

    printf("Latency / throughput:\n");
    printf("  f32 naive:  %.2f ms/matvec   %.2f GFLOP/s\n",  f32_ms, f32_gflops);
    printf("  Q8_0:       %.2f ms/matvec   %.2f GFLOP/s   (%.2fx vs f32)\n",
            q8_ms, q8_gflops, f32_ms / q8_ms);
    printf("  Q4_K_M:     %.2f ms/matvec   %.2f GFLOP/s   (%.2fx vs f32, %.2fx vs Q8_0)\n",
            q4k_ms, q4k_gflops, f32_ms / q4k_ms, q8_ms / q4k_ms);

    free(A_q4k); free(A_q8); free(A_f32); free(x); free(y);
    return 0;
}
