// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// 03_synth_seq — train a tiny transformer to memorize a fixed periodic sequence.
//
// Verifies the full Transformer pipeline end-to-end without requiring external
// data. The "language" here is a 4-character cycle ("ABCD" repeating). A
// well-trained model should reach near-zero loss because the task is
// memorizable; if loss does not drop, autograd somewhere is broken.

#include "slate/slate.h"
#include "slate/transformer.h"
#include "slate/lr_scheduler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VOCAB    4
#define SEQ      16
#define D_MODEL  32
#define N_LAYERS 2
#define FFN_H    64
#define BATCH    8
#define STEPS    400
#define LR       0.01f

int main(void) {
    printf("slate %s — synth_seq (transformer)\n", SLATE_VERSION_STRING);
    slate_arena_t* params  = slate_arena_create( 16 * 1024 * 1024);
    slate_arena_t* opt_st  = slate_arena_create( 16 * 1024 * 1024);
    slate_arena_t* nodes   = slate_arena_create( 16 * 1024 * 1024);
    slate_arena_t* scratch = slate_arena_create(128 * 1024 * 1024);

    slate_module_t* model = slate_module_causal_lm_new(params, VOCAB, SEQ,
                                                        D_MODEL, N_LAYERS, FFN_H,
                                                        1e-5f, 0xC0DEC0DEULL);
    slate_param_set_t ps; slate_param_set_init(&ps);
    slate_module_register_params(model, &ps);
    printf("[synth_seq] %d parameter tensors\n", ps.n_params);
    int64_t total_params = 0;
    for (int i = 0; i < ps.n_params; ++i) total_params += slate_tensor_numel(ps.params[i]);
    printf("[synth_seq] %lld total parameters\n", (long long)total_params);

    slate_optimizer_t* opt = slate_optimizer_adamw_new(opt_st, &ps,
                                                       LR, 0.9f, 0.999f,
                                                       1e-8f, 0.0f);
    slate_lr_scheduler_t* sch = slate_lr_cosine_warmup_new(LR, LR * 0.01f, STEPS / 20, STEPS);

    int64_t xshape[2] = {BATCH, SEQ};
    int64_t yshape[1] = {BATCH * SEQ};
    int64_t logits_flat_shape[2] = {BATCH * SEQ, VOCAB};

    clock_t t0 = clock();
    for (int step = 0; step < STEPS; ++step) {
        slate_optimizer_set_lr(opt, slate_lr_scheduler_get(sch, step));
        slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, nodes, scratch);
        ctx.training = true;

        slate_tensor_t* X = slate_tensor_new(scratch, SLATE_DTYPE_I32, 2, xshape, false);
        slate_tensor_t* Y = slate_tensor_new(scratch, SLATE_DTYPE_I32, 1, yshape, false);
        int32_t* xp = (int32_t*)X->data;
        int32_t* yp = (int32_t*)Y->data;
        // Periodic sequence: token at position t is (t + offset) mod VOCAB.
        for (int b = 0; b < BATCH; ++b) {
            int offset = (step * 7 + b * 3) & 3;
            for (int t = 0; t < SEQ; ++t) {
                xp[b * SEQ + t] = (offset + t) & 3;
                // Target = next token = ((offset + t) + 1) mod VOCAB
                yp[b * SEQ + t] = (offset + t + 1) & 3;
            }
        }
        slate_tensor_t* logits = slate_module_forward(model, &ctx, X);  // [B, T, V]
        // View as [B*T, V] for cross_entropy.
        slate_tensor_t* logits_flat = slate_tensor_view(scratch, logits, 2, logits_flat_shape, NULL);
        slate_tensor_t* loss = slate_op_cross_entropy_loss(&ctx, logits_flat, Y);

        slate_optimizer_zero_grad(opt);
        slate_graph_backward(&ctx, loss);
        slate_clip_grad_norm(&ps, 1.0f);
        slate_optimizer_step(opt);

        if (step == 0 || step == STEPS - 1 || step % 50 == 0) {
            printf("[synth_seq] step %4d  loss=%.4f  lr=%.5f\n",
                   step, ((float*)loss->data)[0], slate_lr_scheduler_get(sch, step));
        }
        slate_graph_ctx_reset(&ctx);
    }
    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("[synth_seq] done in %.2fs\n", secs);

    slate_lr_scheduler_destroy(sch);
    slate_optimizer_destroy(opt);
    slate_param_set_destroy(&ps);
    slate_module_destroy(model);
    slate_arena_destroy(params); slate_arena_destroy(opt_st);
    slate_arena_destroy(nodes); slate_arena_destroy(scratch);
    return 0;
}
