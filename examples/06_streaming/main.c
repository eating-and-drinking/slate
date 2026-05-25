// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
#include "slate/slate.h"
#include "slate/transformer.h"
#include "slate/stream_io.h"
#include "slate/streaming_module.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define D 512
#define N 8

int main(void) {
    printf("slate %s — streaming demo (%d layers x %dx%d)\n", SLATE_VERSION_STRING, N, D, D);
    slate_arena_t* P = slate_arena_create(64 * 1024 * 1024);
    slate_arena_t* Q = slate_arena_create( 4 * 1024 * 1024);
    slate_arena_t* SC = slate_arena_create(128 * 1024 * 1024);
    int64_t ws[2] = {D, D};
    for (int i = 0; i < N; ++i) {
        slate_tensor_t* W = slate_tensor_new(P, SLATE_DTYPE_F32, 2, ws, false);
        uint64_t r = 0x1000 + i;
        for (int64_t k = 0; k < D * D; ++k) {
            r = r * 6364136223846793005ULL + 1442695040888963407ULL;
            ((float*)W->data)[k] = (float)((double)((r >> 11) & ((1ULL<<24)-1)) / (1<<24) - 0.5) * 0.05f;
        }
        char path[64]; snprintf(path, sizeof(path), "/tmp/slate_su_%d.bin", i);
        slate_stream_write(W, path);
    }
    printf("[stream] wrote %d files (%.1f MB each)\n", N, (D * D * 4) / (1024.0 * 1024.0));
    slate_arena_destroy(P);

    slate_module_t* layers[N];
    for (int i = 0; i < N; ++i) {
        char path[64]; snprintf(path, sizeof(path), "/tmp/slate_su_%d.bin", i);
        layers[i] = slate_module_streaming_linear_new(path, D, D);
    }
    printf("[stream] built %d streaming modules; baseline peak = %.2f MB\n",
           N, slate_streaming_peak_bytes() / (1024.0 * 1024.0));

    int64_t xs[2] = {1, D};
    slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, Q, SC); ctx.training = false;
    slate_tensor_t* x = slate_tensor_new(SC, SLATE_DTYPE_F32, 2, xs, false);
    for (int i = 0; i < D; ++i) ((float*)x->data)[i] = 0.1f;
    slate_streaming_reset_peak();
    for (int i = 0; i < N; ++i) x = slate_module_forward(layers[i], &ctx, x);
    printf("[stream] forward through %d layers; final[0]=%.4f\n", N, ((float*)x->data)[0]);

    double peak_mb = slate_streaming_peak_bytes() / (1024.0 * 1024.0);
    double naive_mb = (double)((int64_t)N * D * D * 4) / (1024.0 * 1024.0);
    double one_mb = (double)((int64_t)D * D * 4) / (1024.0 * 1024.0);
    printf("[stream] peak resident during forward = %.2f MB\n", peak_mb);
    printf("[stream] naive (all-resident) = %.2f MB; one-block = %.2f MB\n", naive_mb, one_mb);
    printf("[stream] peak / naive = %.1f%%\n", 100.0 * peak_mb / naive_mb);
    int ok = peak_mb < naive_mb * 0.3;
    for (int i = 0; i < N; ++i) slate_module_destroy(layers[i]);
    slate_arena_destroy(Q); slate_arena_destroy(SC);
    printf("test_streaming: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
