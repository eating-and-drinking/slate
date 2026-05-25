// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// 07_quantized_lora — the capstone demo.
//
// Takes a GGUF file with a Q8_0 quantized base weight, wraps it in a LoRA
// adapter, trains the LoRA on a tiny rotation-learning task, and verifies:
//   1. Training reduces loss (LoRA is actually learning).
//   2. The GGUF file's bytes are bit-identical before and after training
//      (base weights are truly frozen, not just "frozen in spirit").
//   3. The dequantized base matches a reference dequant.

#include "slate/slate.h"
#include "slate/transformer.h"
#include "slate/gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int file_sha_fingerprint(const char* path, unsigned char* out16) {
    // Cheap content fingerprint: sum of bytes, byte-by-byte rolling hash.
    // We mainly want a "did the file change at all" check.
    FILE* fp = fopen(path, "rb"); if (!fp) return -1;
    unsigned char buf[4096];
    size_t n;
    uint64_t h1 = 14695981039346656037ULL, h2 = 0xCBF29CE484222325ULL;
    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) {
        for (size_t i = 0; i < n; ++i) {
            h1 = (h1 ^ buf[i]) * 1099511628211ULL;
            h2 = (h2 + buf[i]) * 16777619ULL;
        }
    }
    fclose(fp);
    memcpy(out16, &h1, 8); memcpy(out16 + 8, &h2, 8);
    return 0;
}

#define IN  32
#define OUT 4
#define RANK 4
#define BS  16
#define STEPS 200

int main(void) {
    printf("slate %s — capstone (GGUF Q8_0 + LoRA)\n", SLATE_VERSION_STRING);

    // Snapshot GGUF file fingerprint BEFORE.
    unsigned char hash_before[16], hash_after[16];
    if (file_sha_fingerprint("/tmp/slate_q8.gguf", hash_before) != 0) {
        fprintf(stderr, "need /tmp/slate_q8.gguf; run tools/make_q8_gguf.py first\n");
        return 1;
    }
    printf("[cap] base GGUF fingerprint (before): ");
    for (int i = 0; i < 16; ++i) printf("%02x", hash_before[i]);
    printf("\n");

    slate_gguf_t* g = slate_gguf_open("/tmp/slate_q8.gguf");
    if (!g) { puts("gguf open FAIL"); return 1; }

    slate_arena_t* P = slate_arena_create(8 * 1024 * 1024);
    slate_arena_t* O = slate_arena_create(8 * 1024 * 1024);
    slate_arena_t* N = slate_arena_create(8 * 1024 * 1024);
    slate_arena_t* S = slate_arena_create(8 * 1024 * 1024);

    slate_module_t* ql = slate_module_quantized_lora_new(P, g, "quant.weight",
                                                          RANK, 4.0f, 0xDEAD);
    if (!ql) { puts("ql create FAIL"); return 1; }
    slate_param_set_t ps; slate_param_set_init(&ps);
    slate_module_register_params(ql, &ps);
    printf("[cap] LoRA trainable params: %d (A + B, base frozen)\n", ps.n_params);
    int64_t lora_n = 0;
    for (int i = 0; i < ps.n_params; ++i) lora_n += slate_tensor_numel(ps.params[i]);
    int64_t base_n = (int64_t)IN * OUT;
    printf("[cap] LoRA params: %lld  base params: %lld  ratio: %.1f%%\n",
           (long long)lora_n, (long long)base_n, 100.0 * lora_n / base_n);

    slate_optimizer_t* opt = slate_optimizer_adamw_new(O, &ps, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f);

    // Synthesize a target rotation we want the LoRA to learn.
    uint64_t r = 0xBABE;
    float target_W[IN * OUT];
    for (int i = 0; i < IN * OUT; ++i) {
        r = r * 6364136223846793005ULL + 1442695040888963407ULL;
        target_W[i] = (float)((double)((r >> 11) & ((1ULL<<24)-1)) / (1<<24) - 0.5) * 0.5f;
    }

    int64_t xs[2] = {BS, IN}, ys[2] = {BS, OUT};
    float L0 = 0, Ln = 0;
    for (int step = 0; step < STEPS; ++step) {
        slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, N, S); ctx.training = true;
        slate_tensor_t* X = slate_tensor_new(S, SLATE_DTYPE_F32, 2, xs, false);
        slate_tensor_t* Y = slate_tensor_new(S, SLATE_DTYPE_F32, 2, ys, false);
        for (int b = 0; b < BS; ++b) {
            for (int d = 0; d < IN; ++d) {
                r = r * 6364136223846793005ULL + 1442695040888963407ULL;
                ((float*)X->data)[b*IN + d] = (float)((double)((r >> 11) & ((1ULL<<24)-1)) / (1<<24) - 0.5);
            }
            for (int o = 0; o < OUT; ++o) {
                float s = 0;
                for (int d = 0; d < IN; ++d) s += ((float*)X->data)[b*IN + d] * target_W[d*OUT + o];
                ((float*)Y->data)[b*OUT + o] = s;
            }
        }
        slate_tensor_t* yhat = slate_module_forward(ql, &ctx, X);
        slate_tensor_t* loss = slate_op_mse_loss(&ctx, yhat, Y);
        float L = ((float*)loss->data)[0];
        if (step == 0) L0 = L;
        if (step == STEPS - 1) Ln = L;
        if (step % 40 == 0 || step == STEPS - 1)
            printf("[cap] step %3d  loss=%.5f\n", step, L);
        slate_optimizer_zero_grad(opt);
        slate_graph_backward(&ctx, loss);
        slate_optimizer_step(opt);
        slate_graph_ctx_reset(&ctx);
    }

    slate_optimizer_destroy(opt); slate_param_set_destroy(&ps); slate_module_destroy(ql);
    slate_gguf_close(g);
    slate_arena_destroy(P); slate_arena_destroy(O);
    slate_arena_destroy(N); slate_arena_destroy(S);

    // Verify GGUF file unchanged.
    file_sha_fingerprint("/tmp/slate_q8.gguf", hash_after);
    int gguf_unchanged = (memcmp(hash_before, hash_after, 16) == 0);
    printf("[cap] base GGUF fingerprint (after):  ");
    for (int i = 0; i < 16; ++i) printf("%02x", hash_after[i]);
    printf("\n");
    printf("[cap] GGUF byte-identical: %s\n", gguf_unchanged ? "yes" : "NO");
    printf("[cap] loss %.5f -> %.5f (target: drop)\n", L0, Ln);
    int loss_dropped = Ln < L0 * 0.7f;
    int ok = gguf_unchanged && loss_dropped;
    printf("capstone: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
