#define _POSIX_C_SOURCE 200809L
// SPDX-License-Identifier: Apache-2.0
// bmm (batched matmul) benchmark — covers the shapes attention actually sees.
//
// In a transformer, two BMMs dominate:
//   Q @ K^T :  [B*H, S, D] @ [B*H, D, S]  ->  [B*H, S, S]
//   attn @ V:  [B*H, S, S] @ [B*H, S, D]  ->  [B*H, S, D]
//
// where B = batch, H = heads, S = seq len, D = head dim.

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

static void bench(int B, int H, int S, int D, int iters, const char* label) {
    slate_arena_t* pa = slate_arena_create(512 * 1024 * 1024);
    slate_arena_t* na = slate_arena_create(  8 * 1024 * 1024);
    slate_arena_t* sa = slate_arena_create(512 * 1024 * 1024);

    int64_t sh_q[3] = {B * H, S, D};
    int64_t sh_k[3] = {B * H, D, S};
    slate_tensor_t* Q = slate_tensor_new(pa, SLATE_DTYPE_F32, 3, sh_q, false);
    slate_tensor_t* K = slate_tensor_new(pa, SLATE_DTYPE_F32, 3, sh_k, false);
    int64_t nQ = (int64_t)B * H * S * D, nK = (int64_t)B * H * D * S;
    for (int64_t i = 0; i < nQ; ++i) ((float*)Q->data)[i] = 0.001f * (float)(i % 997);
    for (int64_t i = 0; i < nK; ++i) ((float*)K->data)[i] = 0.001f * (float)(i % 991);

    // Warmup
    slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, na, sa); ctx.training = false;
    slate_op_bmm(&ctx, Q, K);
    slate_graph_ctx_reset(&ctx);

    double t0 = now_s();
    for (int it = 0; it < iters; ++it) {
        slate_graph_ctx_init(&ctx, na, sa); ctx.training = false;
        slate_op_bmm(&ctx, Q, K);
        slate_graph_ctx_reset(&ctx);
    }
    double dt = now_s() - t0;
    double flops = 2.0 * (double)(B * H) * S * D * S * iters;
    printf("  %-22s [B*H=%3d, S=%3d, D=%3d -> S=%3d]  %.2f ms/iter, %.2f GFLOP/s\n",
           label, B * H, S, D, S, dt * 1000.0 / iters, flops / dt / 1e9);

    slate_arena_destroy(pa); slate_arena_destroy(na); slate_arena_destroy(sa);
}

int main(void) {
    printf("slate bmm (batched matmul) benchmark — %d threads\n",
           slate_threadpool_num_threads(slate_global_pool()));
    printf("(label                  [batch, M, K -> N]   time, throughput)\n");
    // Char-level transformer attention sizes (M2 tinyshakespeare-class)
    bench(8, 4,  64, 32, 100, "char-LM (Q@K^T)");
    // GPT-2 small attention shape: H=12, D=64, S=256
    bench(1, 12, 256, 64, 50, "GPT-2 sm (Q@K^T)");
    bench(1, 12, 256, 64, 50, "GPT-2 sm (attn@V)");
    // Longer context
    bench(1, 8,  512, 64, 20, "GPT-2 sm S=512");
    // Bigger batch
    bench(4, 8,  256, 64, 20, "B=4 H=8 S=256");
    return 0;
}
