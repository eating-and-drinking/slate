#define _POSIX_C_SOURCE 200809L
// SPDX-License-Identifier: Apache-2.0
// bench_softmax_rmsnorm.c — measure softmax + RMSNorm SIMD speedup.
//
// Drives both ops over typical transformer shapes:
//   - softmax over attention scores: row of length S
//   - RMSNorm over hidden dim: row of length d_model

#include "slate/slate.h"
#include "slate/ops.h"
#include "slate/runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void bench_softmax(int N, int C, int iters) {
    slate_arena_t* a = slate_arena_create(64 * 1024 * 1024);
    slate_arena_t* n = slate_arena_create( 4 * 1024 * 1024);
    slate_arena_t* s = slate_arena_create(64 * 1024 * 1024);
    int64_t sh[2] = {N, C};
    slate_tensor_t* x = slate_tensor_new(a, SLATE_DTYPE_F32, 2, sh, false);
    int64_t nx = (int64_t)N * C;
    for (int64_t i = 0; i < nx; ++i) ((float*)x->data)[i] = 0.001f * (float)(i % 1009);
    slate_graph_ctx_t ctx;
    slate_graph_ctx_init(&ctx, n, s); ctx.training = false;
    slate_op_softmax(&ctx, x); slate_graph_ctx_reset(&ctx);  // warmup
    double t0 = now_s();
    for (int it = 0; it < iters; ++it) {
        slate_graph_ctx_init(&ctx, n, s); ctx.training = false;
        slate_op_softmax(&ctx, x);
        slate_graph_ctx_reset(&ctx);
    }
    double dt = now_s() - t0;
    double work = (double)N * C * iters;  // rough proxy for normalised throughput
    printf("  softmax  N=%4d C=%4d : %.3f us/row, %.2f Melems/s\n",
           N, C, dt * 1e6 / (iters * N), work / dt / 1e6);
    slate_arena_destroy(a); slate_arena_destroy(n); slate_arena_destroy(s);
}

static void bench_rmsnorm(int N, int C, int iters) {
    slate_arena_t* a = slate_arena_create(64 * 1024 * 1024);
    slate_arena_t* n = slate_arena_create( 4 * 1024 * 1024);
    slate_arena_t* s = slate_arena_create(64 * 1024 * 1024);
    int64_t sh[2] = {N, C};
    int64_t sw[1] = {C};
    slate_tensor_t* x = slate_tensor_new(a, SLATE_DTYPE_F32, 2, sh, false);
    slate_tensor_t* w = slate_tensor_new(a, SLATE_DTYPE_F32, 1, sw, false);
    int64_t nx = (int64_t)N * C;
    for (int64_t i = 0; i < nx; ++i) ((float*)x->data)[i] = 0.001f * (float)(i % 1009);
    for (int i = 0; i < C; ++i) ((float*)w->data)[i] = 1.0f;
    slate_graph_ctx_t ctx;
    slate_graph_ctx_init(&ctx, n, s); ctx.training = false;
    slate_op_rms_norm(&ctx, x, w, 1e-5f); slate_graph_ctx_reset(&ctx);  // warmup
    double t0 = now_s();
    for (int it = 0; it < iters; ++it) {
        slate_graph_ctx_init(&ctx, n, s); ctx.training = false;
        slate_op_rms_norm(&ctx, x, w, 1e-5f);
        slate_graph_ctx_reset(&ctx);
    }
    double dt = now_s() - t0;
    double work = (double)N * C * iters;
    printf("  rmsnorm  N=%4d C=%4d : %.3f us/row, %.2f Melems/s\n",
           N, C, dt * 1e6 / (iters * N), work / dt / 1e6);
    slate_arena_destroy(a); slate_arena_destroy(n); slate_arena_destroy(s);
}

int main(void) {
    printf("slate softmax + rmsnorm benchmark — %d threads\n",
           slate_threadpool_num_threads(slate_global_pool()));
    // Attention softmax: row length = seq, batched B*H*S rows
    bench_softmax(  32,    8, 5000);  // below SIMD threshold (C<16)
    bench_softmax(8192,  128, 50);    // attention S=128
    bench_softmax(3072,  256, 50);    // attention S=256
    bench_softmax(4096,  512, 20);    // attention S=512
    bench_softmax(  10, 32000, 100);  // final LM head (vocab=32k)
    // RMSNorm: row length = d_model, B*S rows
    bench_rmsnorm( 512,  128, 1000);  // hidden=128
    bench_rmsnorm( 512,  768, 200);   // GPT-2 sm hidden=768
    bench_rmsnorm( 512, 4096, 50);    // LLaMA-7B d_model
    return 0;
}
