// SPDX-License-Identifier: Apache-2.0
#include "slate/slate.h"
#include "slate/gguf.h"
#include "slate/quant.h"
#include "slate/ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    // 1. Open the GGUF and read the Q8_0 tensor view.
    slate_gguf_t* g = slate_gguf_open("/tmp/slate_q8.gguf");
    if (!g) { puts("gguf open FAIL"); return 1; }
    slate_arena_t* meta = slate_arena_create(64 * 1024);
    slate_tensor_t* Wq = slate_gguf_get_tensor(meta, g, "quant.weight");
    if (!Wq) { puts("tensor lookup FAIL"); return 1; }
    printf("[chain] Q8_0 tensor: shape=[%lld,%lld] dtype=%s\n",
           (long long)Wq->shape[0], (long long)Wq->shape[1],
           slate_dtype_name(Wq->dtype));
    int64_t numel = (int64_t)Wq->shape[0] * Wq->shape[1];

    // 2. Dequantize into a fresh f32 tensor.
    slate_arena_t* P = slate_arena_create(16 * 1024 * 1024);
    int64_t Ws[2] = {Wq->shape[0], Wq->shape[1]};
    slate_tensor_t* W = slate_tensor_new(P, SLATE_DTYPE_F32, 2, Ws, false);
    slate_dequant_q8_0((float*)W->data, Wq->data, numel);

    // Compare against Python's expected dequant
    float expected[128];
    FILE* fp = fopen("/tmp/slate_q8_expected.f32", "rb");
    fread(expected, sizeof(float), 128, fp); fclose(fp);
    float max_err = 0;
    for (int i = 0; i < 128; ++i) {
        float e = fabsf(((float*)W->data)[i] - expected[i]);
        if (e > max_err) max_err = e;
    }
    printf("[chain] dequant max abs error vs python: %.6f\n", max_err);
    int dequant_ok = max_err < 1e-3f;

    // 3. Use the dequantized weight in an actual matmul.
    //    y = x [1, 32] @ W [32, 4] -> [1, 4]
    slate_arena_t* N = slate_arena_create(4 * 1024 * 1024);
    slate_arena_t* S = slate_arena_create(4 * 1024 * 1024);
    slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, N, S); ctx.training = false;
    int64_t xs[2] = {1, 32};
    slate_tensor_t* x = slate_tensor_new(S, SLATE_DTYPE_F32, 2, xs, false);
    for (int i = 0; i < 32; ++i) ((float*)x->data)[i] = 1.0f / 32.0f;  // mean
    slate_tensor_t* y = slate_op_matmul(&ctx, x, W);
    printf("[chain] matmul output [1,4]: %.4f %.4f %.4f %.4f\n",
           ((float*)y->data)[0], ((float*)y->data)[1],
           ((float*)y->data)[2], ((float*)y->data)[3]);
    // Sanity: each output is the mean of one column of W
    float means[4] = {0};
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 32; ++r) means[c] += ((float*)W->data)[r * 4 + c];
        means[c] /= 32.0f;
    }
    int matmul_ok = 1;
    for (int c = 0; c < 4; ++c) {
        if (fabsf(((float*)y->data)[c] - means[c]) > 1e-4f) matmul_ok = 0;
    }
    printf("[chain] matmul matches column means: %s\n", matmul_ok ? "yes" : "no");

    slate_arena_destroy(meta); slate_arena_destroy(P);
    slate_arena_destroy(N); slate_arena_destroy(S);
    slate_gguf_close(g);
    int ok = dequant_ok && matmul_ok;
    printf("test_gguf_q8_matmul: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
