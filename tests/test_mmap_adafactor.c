// SPDX-License-Identifier: Apache-2.0
#include "slate/slate.h"
#include "slate/mmap_dataset.h"
#include <stdio.h>
#include <stdlib.h>

slate_optimizer_t* slate_optimizer_adafactor_new(slate_arena_t*, slate_param_set_t*,
                                                  float, float, float, float);

int main(void) {
    // Build a small token stream and write to /tmp.
    int32_t toks[1024];
    for (int i = 0; i < 1024; ++i) toks[i] = i & 0x3FF;
    FILE* fp = fopen("/tmp/slate_mmap_test.bin", "wb");
    fwrite(toks, sizeof(int32_t), 1024, fp); fclose(fp);

    slate_mmap_dataset_t* ds = slate_mmap_open("/tmp/slate_mmap_test.bin");
    if (!ds) { puts("mmap open FAIL"); return 1; }
    printf("mmap n_tokens=%lld\n", (long long)slate_mmap_n_tokens(ds));
    int32_t X[32], Y[32];
    uint64_t rng = 0xABCD;
    int r = slate_mmap_sample_batch(ds, 2, 16, X, Y, &rng);
    printf("mmap sample: rc=%d, X[0]=%d Y[0]=%d (Y should be X+1)\n", r, X[0], Y[0]);
    int ok = (r == 0 && Y[0] == X[0] + 1);
    slate_mmap_close(ds);

    // Adafactor: train a tiny linear regression.
    slate_arena_t* P = slate_arena_create(1024 * 1024);
    slate_arena_t* O = slate_arena_create(1024 * 1024);
    slate_arena_t* N = slate_arena_create(1024 * 1024);
    slate_arena_t* S = slate_arena_create(1024 * 1024);
    int64_t ws[2] = {4, 1};
    slate_tensor_t* W = slate_tensor_new(P, SLATE_DTYPE_F32, 2, ws, true);
    for (int i = 0; i < 4; ++i) ((float*)W->data)[i] = 0.5f;
    slate_param_set_t ps; slate_param_set_init(&ps); slate_param_set_add(&ps, W);
    slate_optimizer_t* opt = slate_optimizer_adafactor_new(O, &ps, 0.05f, 1e-30f, 1e-3f, 1.0f);

    int64_t xs[2] = {8, 4}, ys[2] = {8, 1};
    float L_first = 0, L_last = 0;
    for (int step = 0; step < 200; ++step) {
        slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, N, S); ctx.training = true;
        slate_tensor_t* X = slate_tensor_new(S, SLATE_DTYPE_F32, 2, xs, false);
        slate_tensor_t* Y = slate_tensor_new(S, SLATE_DTYPE_F32, 2, ys, false);
        for (int b = 0; b < 8; ++b) {
            float x0 = (b - 4) * 0.1f;
            for (int d = 0; d < 4; ++d) ((float*)X->data)[b*4 + d] = x0 * (d + 1);
            ((float*)Y->data)[b] = 2.0f * x0;  // target: y = 2*x0
        }
        slate_tensor_t* yhat = slate_op_matmul(&ctx, X, W);
        slate_tensor_t* loss = slate_op_mse_loss(&ctx, yhat, Y);
        float L = ((float*)loss->data)[0];
        if (step == 0) L_first = L; if (step == 199) L_last = L;
        slate_optimizer_zero_grad(opt);
        slate_graph_backward(&ctx, loss);
        slate_optimizer_step(opt);
        slate_graph_ctx_reset(&ctx);
    }
    printf("adafactor: loss %.4f -> %.4f\n", L_first, L_last);
    ok = ok && (L_last < L_first * 0.5f);

    slate_optimizer_destroy(opt); slate_param_set_destroy(&ps);
    slate_arena_destroy(P); slate_arena_destroy(O);
    slate_arena_destroy(N); slate_arena_destroy(S);
    printf("test_mmap_adafactor: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
