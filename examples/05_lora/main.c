// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
#include "slate/slate.h"
#include "slate/transformer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IN  4
#define OUT 4
#define RANK 2
#define BATCH 16
#define STEPS 300

int main(void) {
    printf("slate %s — lora_verify\n", SLATE_VERSION_STRING);
    slate_arena_t* P = slate_arena_create(8 * 1024 * 1024);
    slate_arena_t* O = slate_arena_create(8 * 1024 * 1024);
    slate_arena_t* N = slate_arena_create(8 * 1024 * 1024);
    slate_arena_t* S = slate_arena_create(8 * 1024 * 1024);

    float base_W[IN * OUT];
    float target_W[IN * OUT];
    uint64_t r = 0xABCD;
    for (int i = 0; i < IN * OUT; ++i) {
        r = r * 6364136223846793005ULL + 1442695040888963407ULL;
        base_W[i] = (float)((double)((r >> 11) & ((1ULL<<24)-1)) / (1<<24) - 0.5) * 0.5f;
        r = r * 6364136223846793005ULL + 1442695040888963407ULL;
        target_W[i] = (float)((double)((r >> 11) & ((1ULL<<24)-1)) / (1<<24) - 0.5) * 0.8f;
    }

    slate_module_t* lora = slate_module_lora_new(P, IN, OUT, RANK, 4.0f, base_W, 0xC0DE);
    slate_param_set_t ps; slate_param_set_init(&ps);
    slate_module_register_params(lora, &ps);
    printf("[lora] params registered: %d (should be 2: A and B)\n", ps.n_params);

    typedef struct { slate_module_t base; slate_tensor_t* W_base; slate_tensor_t* A; slate_tensor_t* B;
                     float scale; int rank; } lora_t;
    lora_t* lm = (lora_t*)lora;
    float base_W_before[IN * OUT]; memcpy(base_W_before, lm->W_base->data, sizeof(base_W_before));
    float A_before[IN * RANK]; memcpy(A_before, lm->A->data, sizeof(A_before));
    float B_before[RANK * OUT]; memcpy(B_before, lm->B->data, sizeof(B_before));
    int B_is_zero = 1;
    for (int i = 0; i < RANK * OUT; ++i) if (B_before[i] != 0.0f) { B_is_zero = 0; break; }
    printf("[lora] B init zero: %s\n", B_is_zero ? "yes" : "no");

    slate_optimizer_t* opt = slate_optimizer_adamw_new(O, &ps, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f);
    int64_t xs[2] = {BATCH, IN}, ys[2] = {BATCH, OUT};
    float L0 = 0, Ln = 0;
    for (int step = 0; step < STEPS; ++step) {
        slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, N, S); ctx.training = true;
        slate_tensor_t* X = slate_tensor_new(S, SLATE_DTYPE_F32, 2, xs, false);
        slate_tensor_t* Y = slate_tensor_new(S, SLATE_DTYPE_F32, 2, ys, false);
        for (int b = 0; b < BATCH; ++b) {
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
        slate_tensor_t* yhat = slate_module_forward(lora, &ctx, X);
        slate_tensor_t* loss = slate_op_mse_loss(&ctx, yhat, Y);
        float L = ((float*)loss->data)[0];
        if (step == 0) L0 = L;
        if (step == STEPS - 1) Ln = L;
        if (step % 60 == 0 || step == STEPS - 1)
            printf("[lora] step %3d  loss=%.5f\n", step, L);
        slate_optimizer_zero_grad(opt);
        slate_graph_backward(&ctx, loss);
        slate_optimizer_step(opt);
        slate_graph_ctx_reset(&ctx);
    }

    int base_unchanged = (memcmp(base_W_before, lm->W_base->data, sizeof(base_W_before)) == 0);
    int A_changed = (memcmp(A_before, lm->A->data, sizeof(A_before)) != 0);
    int B_changed = (memcmp(B_before, lm->B->data, sizeof(B_before)) != 0);
    printf("[lora] base unchanged after training: %s\n", base_unchanged ? "yes" : "NO (BUG)");
    printf("[lora] A trained: %s\n", A_changed ? "yes" : "NO (BUG)");
    printf("[lora] B trained: %s\n", B_changed ? "yes" : "NO (BUG)");
    printf("[lora] loss %.5f -> %.5f (target: significant drop)\n", L0, Ln);
    int ok = base_unchanged && A_changed && B_changed && Ln < L0 * 0.5f;

    slate_optimizer_destroy(opt); slate_param_set_destroy(&ps); slate_module_destroy(lora);
    slate_arena_destroy(P); slate_arena_destroy(O); slate_arena_destroy(N); slate_arena_destroy(S);
    printf("test_lora: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
