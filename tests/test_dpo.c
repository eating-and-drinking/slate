// SPDX-License-Identifier: Apache-2.0
// Toy DPO test.
//
// Policy is a single learnable [hidden, vocab] projection over a one-hot
// input. Reference is the same projection initialized to zero (uniform
// distribution). Pairs: "chosen" = token 7, "rejected" = token 2.
//
// After training, the model should assign higher probability to token 7
// (chosen) than to token 2 (rejected) given the same input.

#include "slate/slate.h"
#include "slate/dpo.h"
#include "slate/ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 10
#define HID   8
#define SEQ   3
#define BS    8

int main(void) {
    slate_arena_t* P = slate_arena_create(2 * 1024 * 1024);
    slate_arena_t* O = slate_arena_create(2 * 1024 * 1024);
    slate_arena_t* N = slate_arena_create(2 * 1024 * 1024);
    slate_arena_t* S = slate_arena_create(8 * 1024 * 1024);

    // Policy projection W [HID, VOCAB], trainable.
    int64_t ws[2] = {HID, VOCAB};
    slate_tensor_t* W = slate_tensor_new(P, SLATE_DTYPE_F32, 2, ws, true);
    // Small random init.
    uint64_t r = 0xC0DE;
    for (int i = 0; i < HID * VOCAB; ++i) {
        r = r * 6364136223846793005ULL + 1442695040888963407ULL;
        ((float*)W->data)[i] = (float)((double)((r >> 11) & ((1ULL<<24)-1)) / (1<<24) - 0.5) * 0.1f;
    }
    slate_param_set_t ps; slate_param_set_init(&ps); slate_param_set_add(&ps, W);
    slate_optimizer_t* opt = slate_optimizer_adamw_new(O, &ps, 0.05f, 0.9f, 0.999f, 1e-8f, 0.0f);

    // For DPO we need reference log-probs. Use uniform reference: log(1/V) per token.
    float uniform_lp_per_token = -logf((float)VOCAB);
    float uniform_lp_per_seq = SEQ * uniform_lp_per_token;

    int64_t xs[2] = {BS, HID};  // input: dense [BS, HID] vector
    int64_t cs[2] = {BS, SEQ};  // chosen tokens
    int64_t rs[2] = {BS, SEQ};  // rejected tokens
    int64_t lp_shape[1] = {BS};

    float L0 = 0, Ln = 0;
    int STEPS = 200;
    for (int step = 0; step < STEPS; ++step) {
        slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, N, S); ctx.training = true;

        slate_tensor_t* x = slate_tensor_new(S, SLATE_DTYPE_F32, 2, xs, false);
        // Use a single dense input vector "context"; broadcast over batch.
        for (int b = 0; b < BS; ++b) {
            for (int h = 0; h < HID; ++h) ((float*)x->data)[b*HID + h] = 1.0f / HID;
        }
        // Get logits [BS, HID] @ W [HID, VOCAB] = [BS, VOCAB]; replicate over SEQ.
        slate_tensor_t* logits_2d = slate_op_matmul(&ctx, x, W);
        // View as [BS, 1, VOCAB] then we need [BS, SEQ, VOCAB] by broadcasting
        // — quick & dirty: build the [BS, SEQ, VOCAB] tensor by copy.
        int64_t lshape[3] = {BS, SEQ, VOCAB};
        slate_tensor_t* logits_chosen = slate_tensor_new(S, SLATE_DTYPE_F32, 3, lshape, false);
        slate_tensor_t* logits_rej    = slate_tensor_new(S, SLATE_DTYPE_F32, 3, lshape, false);
        // For autograd to flow, we need this expansion to be in the graph.
        // We'll use a manual broadcast: copy the [BS, VOCAB] logits into both
        // chosen and rejected positions. Since DPO acts on the sums of token
        // logps over SEQ, the gradient flows back through the expansion.
        // Implementation: we'll create the broadcast as part of the graph
        // by allocating a grad buffer and tying it back manually.
        // For this toy, simpler: skip true broadcast — set SEQ=1.

        // Restart with SEQ=1 logic: chosen/rejected each ARE [BS, VOCAB] -> [BS, 1, VOCAB]
        int64_t lshape1[3] = {BS, 1, VOCAB};
        slate_tensor_t* lc = slate_tensor_view(S, logits_2d, 3, lshape1, NULL);
        slate_tensor_t* lr = slate_tensor_view(S, logits_2d, 3, lshape1, NULL);
        slate_tensor_t* tc = slate_tensor_new(S, SLATE_DTYPE_I32, 2, (int64_t[]){BS,1}, false);
        slate_tensor_t* tr = slate_tensor_new(S, SLATE_DTYPE_I32, 2, (int64_t[]){BS,1}, false);
        for (int b = 0; b < BS; ++b) {
            ((int32_t*)tc->data)[b] = 7;  // chosen
            ((int32_t*)tr->data)[b] = 2;  // rejected
        }
        slate_tensor_t* lp_c_ref = slate_tensor_new(S, SLATE_DTYPE_F32, 1, lp_shape, false);
        slate_tensor_t* lp_r_ref = slate_tensor_new(S, SLATE_DTYPE_F32, 1, lp_shape, false);
        for (int b = 0; b < BS; ++b) {
            ((float*)lp_c_ref->data)[b] = uniform_lp_per_token;
            ((float*)lp_r_ref->data)[b] = uniform_lp_per_token;
        }
        (void)logits_chosen; (void)logits_rej; (void)uniform_lp_per_seq;

        slate_tensor_t* loss = slate_op_dpo_loss(&ctx, lc, tc, lp_c_ref,
                                                  lr, tr, lp_r_ref, 0.1f);
        float L = ((float*)loss->data)[0];
        if (step == 0) L0 = L; if (step == STEPS - 1) Ln = L;
        if (step % 40 == 0 || step == STEPS - 1)
            printf("[dpo] step %3d  loss=%.4f\n", step, L);
        slate_optimizer_zero_grad(opt);
        slate_graph_backward(&ctx, loss);
        slate_optimizer_step(opt);
        slate_graph_ctx_reset(&ctx);
    }

    // Inference: check chosen prob vs rejected prob.
    slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, N, S); ctx.training = false;
    slate_tensor_t* x = slate_tensor_new(S, SLATE_DTYPE_F32, 2, (int64_t[]){1, HID}, false);
    for (int h = 0; h < HID; ++h) ((float*)x->data)[h] = 1.0f / HID;
    slate_tensor_t* logits = slate_op_matmul(&ctx, x, W);
    const float* p = (const float*)logits->data;
    float m = p[0]; for (int v = 1; v < VOCAB; ++v) if (p[v] > m) m = p[v];
    double S_ = 0; float probs[VOCAB];
    for (int v = 0; v < VOCAB; ++v) { probs[v] = expf(p[v] - m); S_ += probs[v]; }
    for (int v = 0; v < VOCAB; ++v) probs[v] /= (float)S_;
    printf("[dpo] final probs: ");
    for (int v = 0; v < VOCAB; ++v) printf("%.3f ", probs[v]);
    printf("\n");
    printf("[dpo] P(chosen=7)=%.4f  P(rejected=2)=%.4f\n", probs[7], probs[2]);
    int ok = (Ln < L0 * 0.5f) && (probs[7] > probs[2] * 2.0f);
    printf("[dpo] loss %.4f -> %.4f; chosen > 2x rejected: %s\n",
           L0, Ln, probs[7] > probs[2] * 2.0f ? "yes" : "no");
    slate_graph_ctx_reset(&ctx);

    slate_optimizer_destroy(opt); slate_param_set_destroy(&ps);
    slate_arena_destroy(P); slate_arena_destroy(O); slate_arena_destroy(N); slate_arena_destroy(S);
    printf("test_dpo: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
