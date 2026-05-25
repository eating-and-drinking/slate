// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// Smoke + finite-difference test for a single TransformerBlock end-to-end.
// We verify (a) forward succeeds, (b) loss decreases when we step the optimizer,
// (c) at least one element of the analytic gradient w.r.t. the input matches
// finite differences (full gradcheck across all params would be slow).

#include "slate/slate.h"
#include "slate/transformer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    slate_arena_t* P = slate_arena_create(8 * 1024 * 1024);
    slate_arena_t* O = slate_arena_create(8 * 1024 * 1024);
    slate_arena_t* N = slate_arena_create(8 * 1024 * 1024);
    slate_arena_t* S = slate_arena_create(32 * 1024 * 1024);

    int B = 2, T = 4, D = 8;
    slate_module_t* block = slate_module_transformer_block_new(P, D, 16, 1e-5f, 42);
    slate_param_set_t ps; slate_param_set_init(&ps);
    slate_module_register_params(block, &ps);

    int64_t xs[3] = {B, T, D};
    int64_t ys[3] = {B, T, D};

    // -- Run forward + backward once, snapshot input gradient --
    slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, N, S);
    ctx.training = true;
    slate_tensor_t* x = slate_tensor_new(P, SLATE_DTYPE_F32, 3, xs, true);
    slate_tensor_t* tgt = slate_tensor_new(P, SLATE_DTYPE_F32, 3, ys, false);
    float* xp = (float*)x->data; float* tp = (float*)tgt->data;
    for (int i = 0; i < B*T*D; ++i) {
        xp[i] = 0.03f * (float)(i + 1) - 0.5f;
        tp[i] = 0.02f * (float)(i + 3);
    }
    slate_tensor_zero_grad(x);
    slate_tensor_t* y = slate_module_forward(block, &ctx, x);
    slate_tensor_t* L = slate_op_mse_loss(&ctx, y, tgt);
    float L0 = ((float*)L->data)[0];
    slate_graph_backward(&ctx, L);
    float g_an = ((float*)x->grad)[0];
    slate_graph_ctx_reset(&ctx);

    // -- Finite difference on x[0] --
    float eps = 1e-3f;
    float orig = ((float*)x->data)[0];
    ((float*)x->data)[0] = orig + eps;
    slate_graph_ctx_init(&ctx, N, S); ctx.training = false;
    slate_tensor_t* yp1 = slate_module_forward(block, &ctx, x);
    float Lp = ((float*)slate_op_mse_loss(&ctx, yp1, tgt)->data)[0];
    slate_graph_ctx_reset(&ctx);
    ((float*)x->data)[0] = orig - eps;
    slate_graph_ctx_init(&ctx, N, S); ctx.training = false;
    slate_tensor_t* yp2 = slate_module_forward(block, &ctx, x);
    float Lm = ((float*)slate_op_mse_loss(&ctx, yp2, tgt)->data)[0];
    slate_graph_ctx_reset(&ctx);
    ((float*)x->data)[0] = orig;
    float g_num = (Lp - Lm) / (2.0f * eps);
    float denom = fmaxf(fabsf(g_num), fabsf(g_an));
    float err = denom > 1e-6f ? fabsf(g_num - g_an) / denom : fabsf(g_num - g_an);
    printf("transformer_block input grad[0]:\n");
    printf("  analytic  = %.6f\n  numerical = %.6f\n  rel_err   = %.4f\n", g_an, g_num, err);
    int ok = err < 5e-2f;  // looser tolerance for the deep stack
    printf("  forward loss before step: %.4f\n", L0);

    // -- Optimizer sanity: loss should decrease after a few steps --
    slate_optimizer_t* opt = slate_optimizer_adamw_new(O, &ps, 0.01f, 0.9f, 0.95f, 1e-8f, 0.0f);
    float L_final = L0;
    for (int s = 0; s < 20; ++s) {
        slate_graph_ctx_init(&ctx, N, S); ctx.training = true;
        slate_tensor_t* yy = slate_module_forward(block, &ctx, x);
        slate_tensor_t* LL = slate_op_mse_loss(&ctx, yy, tgt);
        L_final = ((float*)LL->data)[0];
        slate_optimizer_zero_grad(opt);
        slate_graph_backward(&ctx, LL);
        slate_optimizer_step(opt);
        slate_graph_ctx_reset(&ctx);
    }
    printf("  loss after 20 AdamW steps: %.4f\n", L_final);
    ok = ok && (L_final < L0 * 0.95f);

    slate_optimizer_destroy(opt);
    slate_param_set_destroy(&ps);
    slate_module_destroy(block);
    slate_arena_destroy(P); slate_arena_destroy(O); slate_arena_destroy(N); slate_arena_destroy(S);
    printf("%s\n", ok ? "test_transformer_block: OK" : "test_transformer_block: FAIL");
    return ok ? 0 : 1;
}
