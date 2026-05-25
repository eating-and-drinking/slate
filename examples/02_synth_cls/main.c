// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// 02_synth_cls — multi-class classification on a synthetic dataset.
//
// Generates 1024 4-dimensional points and labels them by which quadrant their
// x[0], x[1] coordinates fall in (4-way classification). Trains a small MLP
// with cross-entropy loss + AdamW + cosine LR schedule, and reports accuracy.
//
// This example exists for verification in environments without the MNIST
// data files. It exercises the same pipeline used by 02_mnist.

#include "slate/slate.h"
#include "slate/lr_scheduler.h"
#include "slate/data_simple.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_SAMPLES   1024
#define N_TEST       256
#define INPUT_DIM     4
#define HIDDEN_DIM   32
#define N_CLASSES     4
#define BATCH_SIZE   32
#define N_EPOCHS     20
#define LR_MAX     0.01f
#define LR_MIN     1e-4f

// Deterministic PRNG so the test is repeatable.
static uint64_t rng_state = 0xC0FFEEULL;
static float frand_unit(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((rng_state >> 11) & ((1ULL<<24)-1)) / (float)(1<<24);
}

// 4-class XOR-like task. Each sample is 4 i.i.d. uniform values in [-1, 1].
// The label is determined by TWO independent XOR-style decisions:
//   bit0 = sign(x[0] * x[1]) > 0  (first XOR over dims 0, 1)
//   bit1 = sign(x[2] * x[3]) > 0  (second XOR over dims 2, 3)
//   y = 2*bit1 + bit0  in {0, 1, 2, 3}
// This is NOT linearly separable on any feature; a linear classifier is
// stuck at chance (25%). An MLP with a hidden layer must learn the products
// to solve it. Loss should drop from log(4) ~ 1.386 toward ~0 over training.
static void generate(int n, float* X, int32_t* y) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < INPUT_DIM; ++j) {
            X[i * INPUT_DIM + j] = (frand_unit() - 0.5f) * 2.0f;  // [-1, 1]
        }
        int bit0 = (X[i * INPUT_DIM + 0] * X[i * INPUT_DIM + 1]) > 0.0f;
        int bit1 = (X[i * INPUT_DIM + 2] * X[i * INPUT_DIM + 3]) > 0.0f;
        y[i] = 2 * bit1 + bit0;
    }
}

int main(void) {
    printf("slate %s — synth_cls example\n", SLATE_VERSION_STRING);

    // Generate dataset.
    static float X_train[N_SAMPLES * INPUT_DIM];
    static int32_t y_train[N_SAMPLES];
    static float X_test [N_TEST    * INPUT_DIM];
    static int32_t y_test [N_TEST];
    generate(N_SAMPLES, X_train, y_train);
    generate(N_TEST,    X_test,  y_test);

    // Arenas.
    slate_arena_t* params  = slate_arena_create(8 * 1024 * 1024);
    slate_arena_t* opt_st  = slate_arena_create(8 * 1024 * 1024);
    slate_arena_t* nodes   = slate_arena_create(8 * 1024 * 1024);
    slate_arena_t* scratch = slate_arena_create(8 * 1024 * 1024);

    // Build MLP: 4 -> 32 -> 32 -> 4 (with bias, ReLU between).
    slate_module_t* fc1 = slate_module_linear_new(params, INPUT_DIM,  HIDDEN_DIM, true, 1);
    slate_module_t* fc2 = slate_module_linear_new(params, HIDDEN_DIM, HIDDEN_DIM, true, 2);
    slate_module_t* fc3 = slate_module_linear_new(params, HIDDEN_DIM, N_CLASSES,  true, 3);

    slate_param_set_t ps;
    slate_param_set_init(&ps);
    slate_module_register_params(fc1, &ps);
    slate_module_register_params(fc2, &ps);
    slate_module_register_params(fc3, &ps);
    printf("[synth] %d parameter tensors\n", ps.n_params);

    slate_optimizer_t* opt = slate_optimizer_adamw_new(opt_st, &ps,
                                                       LR_MAX, 0.9f, 0.999f,
                                                       1e-8f, 0.0f);
    int n_batches = N_SAMPLES / BATCH_SIZE;
    int total_steps = n_batches * N_EPOCHS;
    int warmup_steps = total_steps / 20;  // 5%
    slate_lr_scheduler_t* sched = slate_lr_cosine_warmup_new(LR_MAX, LR_MIN,
                                                              warmup_steps,
                                                              total_steps);

    slate_simple_dataloader_t* dl = slate_simple_dataloader_new(N_SAMPLES,
                                                                BATCH_SIZE,
                                                                true, 0xBEEF);
    int batch_idx[BATCH_SIZE];
    int64_t xshape[2] = {BATCH_SIZE, INPUT_DIM};
    int64_t yshape[1] = {BATCH_SIZE};

    int global_step = 0;
    clock_t t0 = clock();
    for (int epoch = 0; epoch < N_EPOCHS; ++epoch) {
        slate_simple_dataloader_reset(dl);
        double epoch_loss = 0;
        int batches_in_epoch = 0;

        while (slate_simple_dataloader_next(dl, batch_idx)) {
            // Set LR for this step.
            float lr = slate_lr_scheduler_get(sched, global_step);
            slate_optimizer_set_lr(opt, lr);

            slate_graph_ctx_t ctx;
            slate_graph_ctx_init(&ctx, nodes, scratch);
            ctx.training = true;

            slate_tensor_t* xb = slate_tensor_new(scratch, SLATE_DTYPE_F32,
                                                   2, xshape, false);
            slate_tensor_t* yb = slate_tensor_new(scratch, SLATE_DTYPE_I32,
                                                   1, yshape, false);
            // Gather.
            float* xb_p = (float*)xb->data;
            int32_t* yb_p = (int32_t*)yb->data;
            for (int i = 0; i < BATCH_SIZE; ++i) {
                memcpy(xb_p + i * INPUT_DIM,
                       X_train + batch_idx[i] * INPUT_DIM,
                       INPUT_DIM * sizeof(float));
                yb_p[i] = y_train[batch_idx[i]];
            }

            // Forward.
            slate_tensor_t* h1 = slate_module_forward(fc1, &ctx, xb);
            slate_tensor_t* a1 = slate_op_relu(&ctx, h1);
            slate_tensor_t* h2 = slate_module_forward(fc2, &ctx, a1);
            slate_tensor_t* a2 = slate_op_relu(&ctx, h2);
            slate_tensor_t* logits = slate_module_forward(fc3, &ctx, a2);
            slate_tensor_t* loss = slate_op_cross_entropy_loss(&ctx, logits, yb);

            epoch_loss += ((const float*)loss->data)[0];
            batches_in_epoch++;

            slate_optimizer_zero_grad(opt);
            slate_graph_backward(&ctx, loss);
            slate_clip_grad_norm(&ps, 1.0f);
            slate_optimizer_step(opt);
            slate_graph_ctx_reset(&ctx);
            global_step++;
        }

        // Eval on test set.
        int correct = 0;
        int n_test_batches = N_TEST / BATCH_SIZE;
        for (int bi = 0; bi < n_test_batches; ++bi) {
            slate_graph_ctx_t ctx;
            slate_graph_ctx_init(&ctx, nodes, scratch);
            ctx.training = false;

            slate_tensor_t* xb = slate_tensor_new(scratch, SLATE_DTYPE_F32,
                                                   2, xshape, false);
            float* xb_p = (float*)xb->data;
            memcpy(xb_p, X_test + bi * BATCH_SIZE * INPUT_DIM,
                   BATCH_SIZE * INPUT_DIM * sizeof(float));

            slate_tensor_t* h1 = slate_module_forward(fc1, &ctx, xb);
            slate_tensor_t* a1 = slate_op_relu(&ctx, h1);
            slate_tensor_t* h2 = slate_module_forward(fc2, &ctx, a1);
            slate_tensor_t* a2 = slate_op_relu(&ctx, h2);
            slate_tensor_t* logits = slate_module_forward(fc3, &ctx, a2);

            const float* p = (const float*)logits->data;
            for (int i = 0; i < BATCH_SIZE; ++i) {
                int best = 0;
                float best_v = p[i * N_CLASSES];
                for (int c = 1; c < N_CLASSES; ++c) {
                    if (p[i * N_CLASSES + c] > best_v) {
                        best_v = p[i * N_CLASSES + c];
                        best = c;
                    }
                }
                if (best == y_test[bi * BATCH_SIZE + i]) correct++;
            }
            slate_graph_ctx_reset(&ctx);
        }
        float acc = (float)correct / (float)(n_test_batches * BATCH_SIZE);
        printf("[synth] epoch %2d  loss=%.4f  test_acc=%.3f  lr=%.5f\n",
               epoch, epoch_loss / batches_in_epoch, acc,
               slate_lr_scheduler_get(sched, global_step - 1));
    }
    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("[synth] done in %.2fs\n", secs);

    slate_simple_dataloader_destroy(dl);
    slate_lr_scheduler_destroy(sched);
    slate_optimizer_destroy(opt);
    slate_param_set_destroy(&ps);
    slate_module_destroy(fc1);
    slate_module_destroy(fc2);
    slate_module_destroy(fc3);
    slate_arena_destroy(params);
    slate_arena_destroy(opt_st);
    slate_arena_destroy(nodes);
    slate_arena_destroy(scratch);
    return 0;
}
_destroy(dl);
    slate_lr_scheduler_destroy(sched);
    slate_optimizer_destroy(opt);
    slate_param_set_destroy(&ps);
    slate_module_destroy(fc1);
    slate_module_destroy(fc2);
    slate_module_destroy(fc3);
    slate_arena_destroy(params);
    slate_arena_destroy(opt_st);
    slate_arena_destroy(nodes);
    slate_arena_destroy(scratch);
    return 0;
}
