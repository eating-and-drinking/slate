// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// 01_xor — the smallest end-to-end Slate program.
//
// Trains a two-layer MLP with sigmoid hidden activations to learn the XOR
// truth table, using AdamW. This program exists to prove the whole stack
// (tensor, autograd, ops, modules, optimizer) wires together correctly. If
// you change a low-level primitive and XOR no longer converges, you have a
// regression.

#include "slate/slate.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_DIM   2
#define HIDDEN_DIM  8
#define OUTPUT_DIM  1
#define BATCH_SIZE  4
#define N_STEPS     2000
#define LR          0.05f

static const float XOR_X[BATCH_SIZE][INPUT_DIM] = {
    {0.f, 0.f}, {0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f},
};
static const float XOR_Y[BATCH_SIZE][OUTPUT_DIM] = {
    {0.f}, {1.f}, {1.f}, {0.f},
};

int main(void) {
    printf("slate %s — xor example\n", SLATE_VERSION_STRING);

    // -------------------------------------------------------------------------
    // Arenas:
    //   param_arena : weights + gradients + optimizer state (persistent)
    //   node_arena  : graph nodes themselves (reset every step)
    //   scratch     : activations and intermediate tensors (reset every step)
    // -------------------------------------------------------------------------
    slate_arena_t* param_arena   = slate_arena_create(16 * 1024 * 1024);
    slate_arena_t* node_arena    = slate_arena_create( 4 * 1024 * 1024);
    slate_arena_t* scratch_arena = slate_arena_create( 4 * 1024 * 1024);
    if (!param_arena || !node_arena || !scratch_arena) {
        fprintf(stderr, "arena allocation failed\n");
        return 1;
    }
    printf("[slate] arena: %zu MiB params, %zu MiB nodes, %zu MiB scratch\n",
           slate_arena_capacity(param_arena)   / (1024 * 1024),
           slate_arena_capacity(node_arena)    / (1024 * 1024),
           slate_arena_capacity(scratch_arena) / (1024 * 1024));

    // -------------------------------------------------------------------------
    // Build the model: Linear(2,8) -> Sigmoid -> Linear(8,1) -> Sigmoid.
    //
    // We disable bias here so we don't need broadcasting support (M1's job).
    // To compensate, we use one extra hidden unit and a sigmoid output, both
    // of which make XOR learnable without a bias.
    // -------------------------------------------------------------------------
    slate_module_t* fc1 = slate_module_linear_new(param_arena,
                                                   INPUT_DIM, HIDDEN_DIM,
                                                   false, 0xC0DEC0DEULL);
    slate_module_t* fc2 = slate_module_linear_new(param_arena,
                                                   HIDDEN_DIM, OUTPUT_DIM,
                                                   false, 0xFEEDFACEULL);
    if (!fc1 || !fc2) { fprintf(stderr, "linear alloc failed\n"); return 1; }

    // We won't use Sequential here because we need to insert activations
    // between linears; instead we write the forward inline below.

    // -------------------------------------------------------------------------
    // Collect parameters and build the optimizer.
    // -------------------------------------------------------------------------
    slate_param_set_t params;
    slate_param_set_init(&params);
    slate_module_register_params(fc1, &params);
    slate_module_register_params(fc2, &params);
    printf("[slate] %d parameter tensors registered\n", params.n_params);

    slate_optimizer_t* opt = slate_optimizer_adamw_new(param_arena, &params,
                                                       LR,
                                                       0.9f, 0.999f, 1e-8f,
                                                       0.0f /* no WD for XOR */);
    if (!opt) { fprintf(stderr, "optimizer alloc failed\n"); return 1; }

    // -------------------------------------------------------------------------
    // Training loop.
    // -------------------------------------------------------------------------
    int64_t xshape[2] = {BATCH_SIZE, INPUT_DIM};
    int64_t yshape[2] = {BATCH_SIZE, OUTPUT_DIM};

    clock_t t0 = clock();
    for (int step = 0; step < N_STEPS; ++step) {
        // Fresh graph context for this step.
        slate_graph_ctx_t ctx;
        slate_graph_ctx_init(&ctx, node_arena, scratch_arena);
        ctx.training = true;

        // Input + target as fresh scratch tensors.
        slate_tensor_t* x = slate_tensor_new(scratch_arena, SLATE_DTYPE_F32,
                                              2, xshape, false);
        slate_tensor_t* y = slate_tensor_new(scratch_arena, SLATE_DTYPE_F32,
                                              2, yshape, false);
        slate_tensor_set_data(x, XOR_X, sizeof(XOR_X));
        slate_tensor_set_data(y, XOR_Y, sizeof(XOR_Y));

        // Forward: fc1 -> sigmoid -> fc2 -> sigmoid.
        slate_tensor_t* h1 = slate_module_forward(fc1, &ctx, x);
        slate_tensor_t* a1 = slate_op_sigmoid(&ctx, h1);
        slate_tensor_t* h2 = slate_module_forward(fc2, &ctx, a1);
        slate_tensor_t* yhat = slate_op_sigmoid(&ctx, h2);

        // Loss.
        slate_tensor_t* loss = slate_op_mse_loss(&ctx, yhat, y);

        // Backward.
        slate_optimizer_zero_grad(opt);
        slate_graph_backward(&ctx, loss);
        slate_optimizer_step(opt);

        if (step % 200 == 0 || step == N_STEPS - 1) {
            float L = ((const float*)loss->data)[0];
            printf("[xor]   step %4d  loss = %.6f\n", step, L);
        }

        // Reset per-step arenas for the next iteration.
        slate_graph_ctx_reset(&ctx);
    }
    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("[xor]   trained MLP on XOR in %.2fs\n", secs);

    // -------------------------------------------------------------------------
    // Final predictions.
    // -------------------------------------------------------------------------
    printf("[xor]   predictions:\n");
    {
        slate_graph_ctx_t ctx;
        slate_graph_ctx_init(&ctx, node_arena, scratch_arena);
        ctx.training = false;

        slate_tensor_t* x = slate_tensor_new(scratch_arena, SLATE_DTYPE_F32,
                                              2, xshape, false);
        slate_tensor_set_data(x, XOR_X, sizeof(XOR_X));

        slate_tensor_t* h1 = slate_module_forward(fc1, &ctx, x);
        slate_tensor_t* a1 = slate_op_sigmoid(&ctx, h1);
        slate_tensor_t* h2 = slate_module_forward(fc2, &ctx, a1);
        slate_tensor_t* yhat = slate_op_sigmoid(&ctx, h2);

        const float* p = (const float*)yhat->data;
        for (int i = 0; i < BATCH_SIZE; ++i) {
            int a = (int)XOR_X[i][0];
            int b = (int)XOR_X[i][1];
            int t = (int)XOR_Y[i][0];
            printf("[xor]     %d XOR %d -> %.3f  (target %d)\n", a, b, p[i], t);
        }
        slate_graph_ctx_reset(&ctx);
    }

    // -------------------------------------------------------------------------
    // Cleanup.
    // -------------------------------------------------------------------------
    slate_optimizer_destroy(opt);
    slate_param_set_destroy(&params);
    slate_module_destroy(fc1);
    slate_module_destroy(fc2);
    slate_arena_destroy(param_arena);
    slate_arena_destroy(node_arena);
    slate_arena_destroy(scratch_arena);
    return 0;
}
