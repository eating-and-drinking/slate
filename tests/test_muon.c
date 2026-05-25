// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// test_muon.c — exercises the Muon optimizer on three things:
//   (1) a 2D-weight linear regression — verifies the NS5 + dim-scale
//       path (and specifically the rows > cols transpose branch);
//   (2) a 2D-weight regression with rows < cols — exercises the
//       non-transposed branch;
//   (3) a 1D bias param — verifies the SGD-with-momentum fallback.
//
// Convergence criterion: loss drops by ≥ 5x over 200 steps.

#include "slate/slate.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Train a tiny linear regression y = X w (+ b)  with Muon.  W has shape
// [in_dim, out_dim].  If `with_bias`, also adds a 1D bias param.
// Returns (loss_first, loss_last) via out params.
static int train_linreg(int in_dim, int out_dim, int with_bias,
                        float lr, float momentum,
                        float* loss_first, float* loss_last) {
    const int batch = 8;
    const int steps = 200;

    slate_arena_t* P = slate_arena_create(1 << 20);
    slate_arena_t* O = slate_arena_create(4 << 20);
    slate_arena_t* N = slate_arena_create(1 << 20);
    slate_arena_t* S = slate_arena_create(1 << 20);

    int64_t ws[2] = { in_dim, out_dim };
    slate_tensor_t* W = slate_tensor_new(P, SLATE_DTYPE_F32, 2, ws, true);
    // Initialise W to small random-ish values (deterministic).
    for (int i = 0; i < in_dim * out_dim; ++i) {
        ((float*)W->data)[i] = ((i * 13 + 7) % 11 - 5) * 0.05f;
    }
    int64_t bs[1] = { out_dim };
    slate_tensor_t* B = NULL;
    if (with_bias) {
        B = slate_tensor_new(P, SLATE_DTYPE_F32, 1, bs, true);
        for (int j = 0; j < out_dim; ++j) ((float*)B->data)[j] = 0.0f;
    }

    slate_param_set_t ps;
    slate_param_set_init(&ps);
    slate_param_set_add(&ps, W);
    if (with_bias) slate_param_set_add(&ps, B);

    slate_optimizer_t* opt = slate_optimizer_muon_new(O, &ps,
                                                      lr, momentum,
                                                      /*weight_decay=*/0.0f,
                                                      /*ns_steps=*/5);
    if (!opt) {
        printf("muon ctor returned NULL\n");
        return 0;
    }

    // True target: each output coord is a fixed-but-distinct linear combo
    // of the inputs, plus optionally a constant bias.
    float W_true[64];   // safe upper bound
    float b_true[8];
    for (int i = 0; i < in_dim * out_dim; ++i) W_true[i] = 0.1f * (i + 1);
    for (int j = 0; j < out_dim; ++j) b_true[j] = (j + 1) * 0.05f;

    int64_t xs[2] = { batch, in_dim };
    int64_t ys[2] = { batch, out_dim };

    *loss_first = 0; *loss_last = 0;
    for (int step = 0; step < steps; ++step) {
        slate_graph_ctx_t ctx;
        slate_graph_ctx_init(&ctx, N, S);
        ctx.training = true;

        slate_tensor_t* X = slate_tensor_new(S, SLATE_DTYPE_F32, 2, xs, false);
        slate_tensor_t* Y = slate_tensor_new(S, SLATE_DTYPE_F32, 2, ys, false);
        for (int b = 0; b < batch; ++b) {
            for (int d = 0; d < in_dim; ++d) {
                ((float*)X->data)[b * in_dim + d] = ((b - 4) * 0.1f) * (d + 1);
            }
            for (int j = 0; j < out_dim; ++j) {
                float yval = b_true[j];
                for (int d = 0; d < in_dim; ++d) {
                    yval += W_true[d * out_dim + j]
                          * ((b - 4) * 0.1f) * (d + 1);
                }
                ((float*)Y->data)[b * out_dim + j] = yval;
            }
        }
        slate_tensor_t* yhat = slate_op_matmul(&ctx, X, W);
        if (with_bias) yhat = slate_op_add_bias(&ctx, yhat, B);
        slate_tensor_t* loss = slate_op_mse_loss(&ctx, yhat, Y);
        float L = ((float*)loss->data)[0];
        if (step == 0)        *loss_first = L;
        if (step == steps-1)  *loss_last  = L;

        slate_optimizer_zero_grad(opt);
        slate_graph_backward(&ctx, loss);
        slate_optimizer_step(opt);
        slate_graph_ctx_reset(&ctx);
    }

    slate_optimizer_destroy(opt);
    slate_param_set_destroy(&ps);
    slate_arena_destroy(P);
    slate_arena_destroy(O);
    slate_arena_destroy(N);
    slate_arena_destroy(S);
    return 1;
}

int main(void) {
    int ok = 1;

    // --- Test 1: tall 2D weight [4,1] — rows > cols, transpose branch. ---
    {
        float L0, L1;
        if (!train_linreg(/*in_dim=*/4, /*out_dim=*/1,
                          /*with_bias=*/0,
                          /*lr=*/0.02f, /*momentum=*/0.95f,
                          &L0, &L1)) {
            ok = 0;
        }
        printf("muon tall [4,1]:  loss  %.4f -> %.4f  (drop %.1fx)\n",
               L0, L1, L1 > 1e-9f ? L0 / L1 : 0.0f);
        if (!(L1 < L0 * 0.2f)) { puts("  FAIL: did not converge"); ok = 0; }
    }

    // --- Test 2: wide 2D weight [2,5] — rows < cols, no transpose. ---
    {
        float L0, L1;
        if (!train_linreg(/*in_dim=*/2, /*out_dim=*/5,
                          /*with_bias=*/0,
                          /*lr=*/0.02f, /*momentum=*/0.95f,
                          &L0, &L1)) {
            ok = 0;
        }
        printf("muon wide [2,5]:  loss  %.4f -> %.4f  (drop %.1fx)\n",
               L0, L1, L1 > 1e-9f ? L0 / L1 : 0.0f);
        if (!(L1 < L0 * 0.2f)) { puts("  FAIL: did not converge"); ok = 0; }
    }

    // --- Test 3: 2D weight + 1D bias — exercises the SGD-momentum fallback. ---
    {
        float L0, L1;
        if (!train_linreg(/*in_dim=*/3, /*out_dim=*/2,
                          /*with_bias=*/1,
                          /*lr=*/0.02f, /*momentum=*/0.95f,
                          &L0, &L1)) {
            ok = 0;
        }
        printf("muon +bias[3,2]:  loss  %.4f -> %.4f  (drop %.1fx)\n",
               L0, L1, L1 > 1e-9f ? L0 / L1 : 0.0f);
        if (!(L1 < L0 * 0.2f)) { puts("  FAIL: did not converge"); ok = 0; }
    }

    printf("test_muon: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
