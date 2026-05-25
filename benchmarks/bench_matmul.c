#define _POSIX_C_SOURCE 200809L
// SPDX-License-Identifier: Apache-2.0
// matmul benchmark: report GFLOP/s at various sizes.

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

static void bench(int M, int K, int N, int iters) {
    slate_arena_t* a = slate_arena_create(256 * 1024 * 1024);
    slate_arena_t* n = slate_arena_create( 16 * 1024 * 1024);
    slate_arena_t* s = slate_arena_create(256 * 1024 * 1024);
    int64_t as[2] = {M, K}, bs[2] = {K, N};
    slate_tensor_t* A = slate_tensor_new(a, SLATE_DTYPE_F32, 2, as, false);
    slate_tensor_t* B = slate_tensor_new(a, SLATE_DTYPE_F32, 2, bs, false);
    int64_t nA = (int64_t)M * K, nB = (int64_t)K * N;
    for (int64_t i = 0; i < nA; ++i) ((float*)A->data)[i] = 0.001f * (float)i;
    for (int64_t i = 0; i < nB; ++i) ((float*)B->data)[i] = 0.001f * (float)i;

    // Warmup
    slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, n, s); ctx.training = false;
    slate_op_matmul(&ctx, A, B); slate_graph_ctx_reset(&ctx);

    double t0 = now_s();
    for (int it = 0; it < iters; ++it) {
        slate_graph_ctx_init(&ctx, n, s); ctx.training = false;
        slate_op_matmul(&ctx, A, B);
        slate_graph_ctx_reset(&ctx);
    }
    double dt = now_s() - t0;
    double flops = 2.0 * M * K * N * iters;
    printf("  %4d x %4d x %4d : %.2f ms/iter, %.2f GFLOP/s\n",
           M, K, N, dt * 1000.0 / iters, flops / dt / 1e9);
    slate_arena_destroy(a); slate_arena_destroy(n); slate_arena_destroy(s);
}

int main(void) {
    printf("slate matmul benchmark — %d threads\n",
           slate_threadpool_num_threads(slate_global_pool()));
    printf("(M x K x N: time per iter, throughput)\n");
    bench(  64,   64,   64, 200);
    bench( 128,  128,  128, 100);
    bench( 256,  256,  256, 50);
    bench( 512,  512,  512, 20);
    bench(1024, 1024, 1024, 5);
    return 0;
}
