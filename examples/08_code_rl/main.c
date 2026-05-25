// SPDX-License-Identifier: Apache-2.0
#include "slate/slate.h"
#include "slate/code_executor.h"
#include "slate/grpo.h"
#include "slate/ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_TEMPLATES 4
#define K 6
#define HID 8
#define STEPS 30

static const char* templates[N_TEMPLATES] = {
    "print(0)",
    "import sys; print(int(sys.stdin.read().split()[0]))",
    "import sys; a,b=map(int,sys.stdin.read().split()); print(a-b)",
    "import sys; a,b=map(int,sys.stdin.read().split()); print(a+b)",
};

typedef struct { const char* input; const char* expected; } testcase_t;
static const testcase_t TESTS[] = {
    {"2 3", "5"}, {"10 20", "30"}, {"100 200", "300"}, {"7 8", "15"},
};
#define N_TESTS 4

static float run_template_reward(slate_executor_t* e, int idx) {
    slate_exec_limits_t lim = {2000, 128, 0};
    int passed = 0;
    for (int t = 0; t < N_TESTS; ++t) {
        slate_exec_result_t r;
        slate_executor_run(e, templates[idx], TESTS[t].input, &lim, &r);
        if (r.exit_code == 0 && strstr(r.stdout_data, TESTS[t].expected) != NULL) passed++;
        slate_exec_result_free(&r);
    }
    return (float)passed / (float)N_TESTS;
}

int main(void) {
    printf("slate %s — code RL with sandbox\n", SLATE_VERSION_STRING);
    slate_executor_t* exec = slate_executor_subprocess_new("python3");
    printf("[crl] baseline rewards per template:\n");
    for (int i = 0; i < N_TEMPLATES; ++i)
        printf("[crl]   [%d] reward=%.2f  code=%s\n", i, run_template_reward(exec, i), templates[i]);

    slate_arena_t* P = slate_arena_create(2*1024*1024);
    slate_arena_t* O = slate_arena_create(2*1024*1024);
    slate_arena_t* N = slate_arena_create(4*1024*1024);
    slate_arena_t* S = slate_arena_create(8*1024*1024);
    int64_t ws[2] = {HID, N_TEMPLATES};
    slate_tensor_t* W = slate_tensor_new(P, SLATE_DTYPE_F32, 2, ws, true);
    uint64_t r = 0x12345;
    for (int i = 0; i < HID * N_TEMPLATES; ++i) {
        r = r * 6364136223846793005ULL + 1442695040888963407ULL;
        ((float*)W->data)[i] = (float)((double)((r >> 11) & ((1ULL<<24)-1)) / (1<<24) - 0.5) * 0.01f;
    }
    slate_param_set_t ps; slate_param_set_init(&ps); slate_param_set_add(&ps, W);
    slate_optimizer_t* opt = slate_optimizer_adamw_new(O, &ps, 0.1f, 0.9f, 0.999f, 1e-8f, 0.0f);
    float R0 = 0, Rn = 0;
    for (int step = 0; step < STEPS; ++step) {
        slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, N, S); ctx.training = true;
        slate_tensor_t* x = slate_tensor_new(S, SLATE_DTYPE_F32, 2, (int64_t[]){1, HID}, false);
        for (int h = 0; h < HID; ++h) ((float*)x->data)[h] = 1.0f / HID;
        slate_tensor_t* logits_1d = slate_op_matmul(&ctx, x, W);
        const float* pl = (const float*)logits_1d->data;
        float probs[N_TEMPLATES];
        float m = pl[0]; for (int v = 1; v < N_TEMPLATES; ++v) if (pl[v] > m) m = pl[v];
        double Su = 0; for (int v = 0; v < N_TEMPLATES; ++v) { probs[v] = expf(pl[v] - m); Su += probs[v]; }
        for (int v = 0; v < N_TEMPLATES; ++v) probs[v] /= (float)Su;

        slate_tensor_t* x_tile = slate_tensor_new(S, SLATE_DTYPE_F32, 2, (int64_t[]){K, HID}, false);
        for (int k = 0; k < K; ++k) memcpy((float*)x_tile->data + k*HID, x->data, HID*sizeof(float));
        slate_tensor_t* logits_flat = slate_op_matmul(&ctx, x_tile, W);
        slate_tensor_t* logits = slate_tensor_view(S, logits_flat, 3, (int64_t[]){K, 1, N_TEMPLATES}, NULL);
        slate_tensor_t* tgts = slate_tensor_new(S, SLATE_DTYPE_I32, 2, (int64_t[]){K, 1}, false);
        slate_tensor_t* rew  = slate_tensor_new(S, SLATE_DTYPE_F32, 1, (int64_t[]){K}, false);
        float sum_R = 0;
        for (int k = 0; k < K; ++k) {
            r = r * 6364136223846793005ULL + 1442695040888963407ULL;
            float u = (float)((double)((r >> 11) & ((1ULL<<24)-1)) / (1<<24));
            float c = 0; int choice = N_TEMPLATES - 1;
            for (int v = 0; v < N_TEMPLATES; ++v) { c += probs[v]; if (u <= c) { choice = v; break; } }
            ((int32_t*)tgts->data)[k] = choice;
            float Rk = run_template_reward(exec, choice);
            ((float*)rew->data)[k] = Rk;
            sum_R += Rk;
        }
        float mean_R = sum_R / (float)K;
        if (step == 0) R0 = mean_R;
        if (step == STEPS - 1) Rn = mean_R;
        slate_grpo_config_t cfg = {1, 1, 1};
        slate_tensor_t* loss = slate_op_grpo_loss(&ctx, logits, tgts, rew, &cfg);
        slate_optimizer_zero_grad(opt);
        slate_graph_backward(&ctx, loss);
        slate_clip_grad_norm(&ps, 1.0f);
        slate_optimizer_step(opt);
        if (step % 5 == 0 || step == STEPS - 1)
            printf("[crl] step %2d  mean_R=%.3f  loss=%.3f  probs=[%.2f %.2f %.2f %.2f]\n",
                   step, mean_R, ((float*)loss->data)[0], probs[0], probs[1], probs[2], probs[3]);
        slate_graph_ctx_reset(&ctx);
    }
    slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, N, S); ctx.training = false;
    slate_tensor_t* x = slate_tensor_new(S, SLATE_DTYPE_F32, 2, (int64_t[]){1, HID}, false);
    for (int h = 0; h < HID; ++h) ((float*)x->data)[h] = 1.0f / HID;
    slate_tensor_t* logits = slate_op_matmul(&ctx, x, W);
    const float* pl = (const float*)logits->data;
    float m = pl[0]; for (int v = 1; v < N_TEMPLATES; ++v) if (pl[v] > m) m = pl[v];
    double Su = 0; float pf[N_TEMPLATES];
    for (int v = 0; v < N_TEMPLATES; ++v) { pf[v] = expf(pl[v] - m); Su += pf[v]; }
    for (int v = 0; v < N_TEMPLATES; ++v) pf[v] /= (float)Su;
    printf("[crl] final probs: %.3f %.3f %.3f %.3f\n", pf[0], pf[1], pf[2], pf[3]);
    printf("[crl] mean reward %.3f -> %.3f\n", R0, Rn);
    int ok = (Rn > 0.7f) && (pf[3] > 0.6f);
    printf("code_rl: %s\n", ok ? "OK" : "FAIL");
    slate_optimizer_destroy(opt); slate_param_set_destroy(&ps);
    slate_arena_destroy(P); slate_arena_destroy(O); slate_arena_destroy(N); slate_arena_destroy(S);
    slate_executor_destroy(exec);
    return ok ? 0 : 1;
}
