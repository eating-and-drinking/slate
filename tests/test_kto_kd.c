// SPDX-License-Identifier: Apache-2.0
// KTO + KD toy verification.

#include "slate/slate.h"
#include "slate/kto.h"
#include "slate/kd.h"
#include "slate/ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define V 10
#define HID 8
#define BS 16

static float softmax_prob(const float* logits, int vocab, int idx) {
    float m = logits[0]; for (int i = 1; i < vocab; ++i) if (logits[i] > m) m = logits[i];
    double S = 0; for (int i = 0; i < vocab; ++i) S += expf(logits[i] - m);
    return (float)(expf(logits[idx] - m) / S);
}

int main(void) {
    int all_ok = 1;
    // ============================================================
    // PART 1: KTO with mostly +1 (good) labels on token 5
    // ============================================================
    {
        slate_arena_t* P = slate_arena_create(2*1024*1024);
        slate_arena_t* O = slate_arena_create(2*1024*1024);
        slate_arena_t* N = slate_arena_create(2*1024*1024);
        slate_arena_t* S = slate_arena_create(4*1024*1024);
        int64_t ws[2] = {HID, V};
        slate_tensor_t* W = slate_tensor_new(P, SLATE_DTYPE_F32, 2, ws, true);
        uint64_t r = 0xC0DE;
        for (int i = 0; i < HID*V; ++i) {
            r = r * 6364136223846793005ULL + 1442695040888963407ULL;
            ((float*)W->data)[i] = (float)((double)((r>>11)&((1ULL<<24)-1))/(1<<24) - 0.5) * 0.1f;
        }
        slate_param_set_t ps; slate_param_set_init(&ps); slate_param_set_add(&ps, W);
        slate_optimizer_t* opt = slate_optimizer_adamw_new(O, &ps, 0.05f, 0.9f, 0.999f, 1e-8f, 0.0f);

        float uniform_lp = -logf((float)V);
        for (int step = 0; step < 200; ++step) {
            slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, N, S); ctx.training = true;
            slate_tensor_t* x = slate_tensor_new(S, SLATE_DTYPE_F32, 2, (int64_t[]){BS, HID}, false);
            for (int b = 0; b < BS; ++b) for (int h = 0; h < HID; ++h) ((float*)x->data)[b*HID+h] = 1.0f/HID;
            slate_tensor_t* logits2d = slate_op_matmul(&ctx, x, W);
            int64_t l3[3] = {BS, 1, V};
            slate_tensor_t* logits = slate_tensor_view(S, logits2d, 3, l3, NULL);
            slate_tensor_t* tgts = slate_tensor_new(S, SLATE_DTYPE_I32, 2, (int64_t[]){BS,1}, false);
            slate_tensor_t* lref = slate_tensor_new(S, SLATE_DTYPE_F32, 1, (int64_t[]){BS}, false);
            slate_tensor_t* lbl = slate_tensor_new(S, SLATE_DTYPE_I32, 1, (int64_t[]){BS}, false);
            for (int b = 0; b < BS; ++b) {
                ((int32_t*)tgts->data)[b] = (b < 12) ? 5 : 1;   // 12 sample target token 5 (good), 4 target token 1 (bad)
                ((float*)lref->data)[b] = uniform_lp;
                ((int32_t*)lbl->data)[b] = (b < 12) ? +1 : -1;
            }
            slate_tensor_t* loss = slate_op_kto_loss(&ctx, logits, tgts, lref, lbl,
                                                      0.1f, 1.0f, 1.0f);
            slate_optimizer_zero_grad(opt);
            slate_graph_backward(&ctx, loss);
            slate_optimizer_step(opt);
            if (step % 50 == 0 || step == 199)
                printf("[kto] step %3d  loss=%.4f\n", step, ((float*)loss->data)[0]);
            slate_graph_ctx_reset(&ctx);
        }
        // Inference
        slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, N, S); ctx.training = false;
        slate_tensor_t* x = slate_tensor_new(S, SLATE_DTYPE_F32, 2, (int64_t[]){1, HID}, false);
        for (int h = 0; h < HID; ++h) ((float*)x->data)[h] = 1.0f/HID;
        slate_tensor_t* logits = slate_op_matmul(&ctx, x, W);
        float p5 = softmax_prob((float*)logits->data, V, 5);
        float p1 = softmax_prob((float*)logits->data, V, 1);
        printf("[kto] P(good=5)=%.4f  P(bad=1)=%.4f  -> %s\n",
               p5, p1, p5 > p1 ? "good > bad" : "FAIL");
        all_ok = all_ok && (p5 > p1);
        slate_optimizer_destroy(opt); slate_param_set_destroy(&ps);
        slate_arena_destroy(P); slate_arena_destroy(O); slate_arena_destroy(N); slate_arena_destroy(S);
    }

    // ============================================================
    // PART 2: KD — student should learn teacher's distribution
    // ============================================================
    {
        slate_arena_t* P = slate_arena_create(2*1024*1024);
        slate_arena_t* O = slate_arena_create(2*1024*1024);
        slate_arena_t* N = slate_arena_create(2*1024*1024);
        slate_arena_t* S = slate_arena_create(4*1024*1024);

        // Student logits as trainable params.
        int64_t ls[3] = {2, 3, V};
        slate_tensor_t* student_W = slate_tensor_new(P, SLATE_DTYPE_F32, 3, ls, true);
        for (int i = 0; i < 2*3*V; ++i) ((float*)student_W->data)[i] = 0.0f;  // uniform
        // Teacher logits: 0 everywhere except peak at token 3 (high confidence)
        slate_tensor_t* teacher = slate_tensor_new(P, SLATE_DTYPE_F32, 3, ls, false);
        for (int i = 0; i < 2*3*V; ++i) ((float*)teacher->data)[i] = 0.0f;
        for (int bt = 0; bt < 6; ++bt) ((float*)teacher->data)[bt*V + 3] = 5.0f;

        slate_param_set_t ps; slate_param_set_init(&ps); slate_param_set_add(&ps, student_W);
        slate_optimizer_t* opt = slate_optimizer_adamw_new(O, &ps, 0.05f, 0.9f, 0.999f, 1e-8f, 0.0f);

        for (int step = 0; step < 200; ++step) {
            slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, N, S); ctx.training = true;
            slate_tensor_t* loss = slate_op_kd_loss(&ctx, student_W, teacher, 2.0f);
            if (step % 50 == 0 || step == 199)
                printf("[kd]  step %3d  loss=%.4f\n", step, ((float*)loss->data)[0]);
            slate_optimizer_zero_grad(opt);
            slate_graph_backward(&ctx, loss);
            slate_optimizer_step(opt);
            slate_graph_ctx_reset(&ctx);
        }
        // Check: student's first position should now have token 3 as the peak.
        float* sl = (float*)student_W->data;
        int max_idx = 0; for (int v = 1; v < V; ++v) if (sl[v] > sl[max_idx]) max_idx = v;
        printf("[kd]  student peak: token %d (teacher peak: token 3) -> %s\n",
               max_idx, max_idx == 3 ? "match" : "FAIL");
        all_ok = all_ok && (max_idx == 3);
        slate_optimizer_destroy(opt); slate_param_set_destroy(&ps);
        slate_arena_destroy(P); slate_arena_destroy(O); slate_arena_destroy(N); slate_arena_destroy(S);
    }

    printf("test_kto_kd: %s\n", all_ok ? "OK" : "FAIL");
    return all_ok ? 0 : 1;
}
