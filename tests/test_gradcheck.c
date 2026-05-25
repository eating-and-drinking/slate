// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// test_gradcheck — finite-difference validation of every op's backward.
//
// For each op we:
//   1. Build a tiny graph that ends in a scalar via mse_loss.
//   2. Compute the analytic gradient via backward.
//   3. Compute the numerical gradient by perturbing each input element by
//      ±epsilon and observing the loss difference.
//   4. Compare; declare success if max relative error < TOL.
//
// This is the single most important regression test in the project. If you
// add a new op, you add a case here. Period.

#include "slate/slate.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOL  1e-2f
#define EPS  1e-3f

// Helper: scalar loss given a forward function and current input data.
typedef slate_tensor_t* (*forward_fn)(slate_graph_ctx_t* ctx,
                                       slate_tensor_t* x,
                                       slate_tensor_t* aux,
                                       slate_tensor_t* target);

static float eval_loss(slate_arena_t* nodes,
                        slate_arena_t* scratch,
                        forward_fn fn,
                        slate_tensor_t* x_owner,
                        slate_tensor_t* aux,
                        slate_tensor_t* target) {
    slate_graph_ctx_t ctx;
    slate_graph_ctx_init(&ctx, nodes, scratch);
    ctx.training = false;
    slate_tensor_t* y = fn(&ctx, x_owner, aux, target);
    slate_tensor_t* loss = slate_op_mse_loss(&ctx, y, target);
    float L = ((float*)loss->data)[0];
    slate_graph_ctx_reset(&ctx);
    return L;
}

static int check_op(const char* name, forward_fn fn,
                     int64_t* xshape, int xdims,
                     int64_t* aux_shape, int aux_dims) {
    slate_arena_t* params = slate_arena_create(1024 * 1024);
    slate_arena_t* nodes  = slate_arena_create(1024 * 1024);
    slate_arena_t* scrat  = slate_arena_create(1024 * 1024);

    slate_tensor_t* x = slate_tensor_new(params, SLATE_DTYPE_F32, xdims, xshape, true);
    slate_tensor_t* aux = aux_shape
        ? slate_tensor_new(params, SLATE_DTYPE_F32, aux_dims, aux_shape, true)
        : NULL;
    slate_tensor_t* tgt = slate_tensor_new(params, SLATE_DTYPE_F32, xdims, xshape, false);
    // Fill with deterministic but non-trivial values.
    int64_t nx = slate_tensor_numel(x);
    for (int64_t i = 0; i < nx; ++i) {
        // Deliberately avoid exactly zero to keep ReLU FD-gradient sane.
        ((float*)x->data)[i] = 0.1f * (float)(i + 1) - 0.35f;
    }
    if (aux) {
        int64_t na = slate_tensor_numel(aux);
        for (int64_t i = 0; i < na; ++i) {
            ((float*)aux->data)[i] = 0.07f * (float)(i + 1) - 0.2f;
        }
    }
    int64_t nt = slate_tensor_numel(tgt);
    for (int64_t i = 0; i < nt; ++i) {
        ((float*)tgt->data)[i] = 0.05f * (float)(i + 2);
    }

    // -------- analytic gradient via backward --------
    slate_tensor_zero_grad(x);
    if (aux) slate_tensor_zero_grad(aux);

    slate_graph_ctx_t ctx;
    slate_graph_ctx_init(&ctx, nodes, scrat);
    ctx.training = true;
    slate_tensor_t* y = fn(&ctx, x, aux, tgt);
    slate_tensor_t* loss = slate_op_mse_loss(&ctx, y, tgt);
    slate_graph_backward(&ctx, loss);

    // Snapshot the analytic grad before we reset arenas.
    float* g_analytic = (float*)malloc((size_t)nx * sizeof(float));
    memcpy(g_analytic, x->grad, (size_t)nx * sizeof(float));
    slate_graph_ctx_reset(&ctx);

    // -------- numerical gradient via centered finite differences --------
    int ok = 1;
    float max_err = 0;
    for (int64_t i = 0; i < nx; ++i) {
        float orig = ((float*)x->data)[i];
        ((float*)x->data)[i] = orig + EPS;
        float Lp = eval_loss(nodes, scrat, fn, x, aux, tgt);
        ((float*)x->data)[i] = orig - EPS;
        float Lm = eval_loss(nodes, scrat, fn, x, aux, tgt);
        ((float*)x->data)[i] = orig;

        float g_num = (Lp - Lm) / (2.0f * EPS);
        float g_an  = g_analytic[i];
        float denom = fmaxf(fabsf(g_num), fabsf(g_an));
        float err = denom > 1e-6f ? fabsf(g_num - g_an) / denom : fabsf(g_num - g_an);
        if (err > max_err) max_err = err;
        if (err > TOL) {
            fprintf(stderr, "  %s[%lld]: analytic=%.6f numeric=%.6f err=%.4f\n",
                    name, (long long)i, g_an, g_num, err);
            ok = 0;
        }
    }

    free(g_analytic);
    slate_arena_destroy(params);
    slate_arena_destroy(nodes);
    slate_arena_destroy(scrat);

    printf("  %-16s max_err=%.2e  %s\n", name, max_err, ok ? "OK" : "FAIL");
    return ok;
}

// --- per-op forward wrappers ---
static slate_tensor_t* fw_add(slate_graph_ctx_t* ctx,
                               slate_tensor_t* x, slate_tensor_t* aux,
                               slate_tensor_t* t) {
    (void)t;
    return slate_op_add(ctx, x, aux);
}
static slate_tensor_t* fw_mul(slate_graph_ctx_t* ctx,
                               slate_tensor_t* x, slate_tensor_t* aux,
                               slate_tensor_t* t) {
    (void)t;
    return slate_op_mul(ctx, x, aux);
}
static slate_tensor_t* fw_relu(slate_graph_ctx_t* ctx,
                                slate_tensor_t* x, slate_tensor_t* aux,
                                slate_tensor_t* t) {
    (void)aux; (void)t;
    return slate_op_relu(ctx, x);
}
static slate_tensor_t* fw_sigmoid(slate_graph_ctx_t* ctx,
                                   slate_tensor_t* x, slate_tensor_t* aux,
                                   slate_tensor_t* t) {
    (void)aux; (void)t;
    return slate_op_sigmoid(ctx, x);
}
static slate_tensor_t* fw_softmax(slate_graph_ctx_t* ctx,
                                   slate_tensor_t* x, slate_tensor_t* aux,
                                   slate_tensor_t* t) {
    (void)aux; (void)t;
    return slate_op_softmax(ctx, x);
}
static slate_tensor_t* fw_add_bias(slate_graph_ctx_t* ctx,
                                    slate_tensor_t* x, slate_tensor_t* aux,
                                    slate_tensor_t* t) {
    (void)t;
    return slate_op_add_bias(ctx, x, aux);
}
// (matmul uses a custom driver inside main() since its output shape
//  differs from input shape.)

int main(void) {
    int all_ok = 1;
    int64_t s2x3[2] = {2, 3};
    int64_t s3x2[2] = {3, 2};
    int64_t s2x2[2] = {2, 2};

    printf("gradcheck:\n");
    all_ok &= check_op("add",     fw_add,     s2x3, 2, s2x3, 2);
    all_ok &= check_op("mul",     fw_mul,     s2x3, 2, s2x3, 2);
    all_ok &= check_op("relu",    fw_relu,    s2x3, 2, NULL, 0);
    all_ok &= check_op("sigmoid", fw_sigmoid, s2x3, 2, NULL, 0);
    all_ok &= check_op("softmax", fw_softmax, s2x3, 2, NULL, 0);
    {
        int64_t bs[1] = {3};
        all_ok &= check_op("add_bias", fw_add_bias, s2x3, 2, bs, 1);
    }
    // cross_entropy needs a non-trivial custom driver (i32 targets).
    {
        int64_t logits_shape[2] = {4, 3};
        int64_t tgt_shape[1] = {4};
        slate_arena_t* params = slate_arena_create(1024 * 1024);
        slate_arena_t* nodes  = slate_arena_create(1024 * 1024);
        slate_arena_t* scrat  = slate_arena_create(1024 * 1024);
        slate_tensor_t* logits = slate_tensor_new(params, SLATE_DTYPE_F32, 2, logits_shape, true);
        slate_tensor_t* tgts   = slate_tensor_new(params, SLATE_DTYPE_I32, 1, tgt_shape,   false);
        for (int i = 0; i < 12; ++i) ((float*)logits->data)[i] = 0.1f * (i + 1) - 0.5f;
        int32_t tv[4] = {0, 1, 2, 1};
        memcpy(tgts->data, tv, sizeof(tv));

        slate_graph_ctx_t ctx;
        slate_graph_ctx_init(&ctx, nodes, scrat);
        ctx.training = true;
        slate_tensor_t* loss = slate_op_cross_entropy_loss(&ctx, logits, tgts);
        slate_graph_backward(&ctx, loss);
        float g_an[12]; memcpy(g_an, logits->grad, sizeof(g_an));
        slate_graph_ctx_reset(&ctx);

        int ok = 1;
        float max_err = 0;
        for (int i = 0; i < 12; ++i) {
            float orig = ((float*)logits->data)[i];
            ((float*)logits->data)[i] = orig + EPS;
            slate_graph_ctx_init(&ctx, nodes, scrat); ctx.training = false;
            slate_tensor_t* L1 = slate_op_cross_entropy_loss(&ctx, logits, tgts);
            float vp = ((float*)L1->data)[0];
            slate_graph_ctx_reset(&ctx);

            ((float*)logits->data)[i] = orig - EPS;
            slate_graph_ctx_init(&ctx, nodes, scrat); ctx.training = false;
            slate_tensor_t* L2 = slate_op_cross_entropy_loss(&ctx, logits, tgts);
            float vm = ((float*)L2->data)[0];
            slate_graph_ctx_reset(&ctx);

            ((float*)logits->data)[i] = orig;
            float g_num = (vp - vm) / (2 * EPS);
            float denom = fmaxf(fabsf(g_num), fabsf(g_an[i]));
            float err = denom > 1e-6f ? fabsf(g_num - g_an[i]) / denom
                                      : fabsf(g_num - g_an[i]);
            if (err > max_err) max_err = err;
            if (err > TOL) ok = 0;
        }
        printf("  %-16s max_err=%.2e  %s\n", "cross_entropy", max_err, ok ? "OK" : "FAIL");
        all_ok &= ok;
        slate_arena_destroy(params);
        slate_arena_destroy(nodes);
        slate_arena_destroy(scrat);
    }
    // matmul: x[2,3] @ b[3,2] -> [2,2]; output shape differs from input shape,
    // so we can't reuse check_op directly. Custom mini-driver.
    {
        int64_t s2x3[2] = {2, 3};
        int64_t s3x2[2] = {3, 2};
        int64_t s2x2[2] = {2, 2};
        slate_arena_t* params = slate_arena_create(1024 * 1024);
        slate_arena_t* nodes  = slate_arena_create(1024 * 1024);
        slate_arena_t* scrat  = slate_arena_create(1024 * 1024);
        slate_tensor_t* x = slate_tensor_new(params, SLATE_DTYPE_F32, 2, s2x3, true);
        slate_tensor_t* b = slate_tensor_new(params, SLATE_DTYPE_F32, 2, s3x2, true);
        slate_tensor_t* t = slate_tensor_new(params, SLATE_DTYPE_F32, 2, s2x2, false);
        for (int i = 0; i < 6; ++i) ((float*)x->data)[i] = 0.1f * (i + 1) - 0.35f;
        for (int i = 0; i < 6; ++i) ((float*)b->data)[i] = 0.07f * (i + 1) - 0.2f;
        for (int i = 0; i < 4; ++i) ((float*)t->data)[i] = 0.05f * (i + 2);

        slate_graph_ctx_t ctx;
        slate_graph_ctx_init(&ctx, nodes, scrat);
        ctx.training = true;
        slate_tensor_t* y = slate_op_matmul(&ctx, x, b);
        slate_tensor_t* loss = slate_op_mse_loss(&ctx, y, t);
        slate_graph_backward(&ctx, loss);
        float g_an[6]; memcpy(g_an, x->grad, sizeof(g_an));
        slate_graph_ctx_reset(&ctx);

        int ok = 1; float max_err = 0;
        for (int i = 0; i < 6; ++i) {
            float orig = ((float*)x->data)[i];
            ((float*)x->data)[i] = orig + EPS;
            slate_graph_ctx_init(&ctx, nodes, scrat); ctx.training = false;
            slate_tensor_t* y1 = slate_op_matmul(&ctx, x, b);
            slate_tensor_t* L1 = slate_op_mse_loss(&ctx, y1, t);
            float vp = ((float*)L1->data)[0];
            slate_graph_ctx_reset(&ctx);
            ((float*)x->data)[i] = orig - EPS;
            slate_graph_ctx_init(&ctx, nodes, scrat); ctx.training = false;
            slate_tensor_t* y2 = slate_op_matmul(&ctx, x, b);
            slate_tensor_t* L2 = slate_op_mse_loss(&ctx, y2, t);
            float vm = ((float*)L2->data)[0];
            slate_graph_ctx_reset(&ctx);
            ((float*)x->data)[i] = orig;
            float g_num = (vp - vm) / (2 * EPS);
            float denom = fmaxf(fabsf(g_num), fabsf(g_an[i]));
            float err = denom > 1e-6f ? fabsf(g_num - g_an[i]) / denom
                                      : fabsf(g_num - g_an[i]);
            if (err > max_err) max_err = err;
            if (err > TOL) ok = 0;
        }
        printf("  %-16s max_err=%.2e  %s\n", "matmul", max_err, ok ? "OK" : "FAIL");
        all_ok &= ok;
        slate_arena_destroy(params);
        slate_arena_destroy(nodes);
        slate_arena_destroy(scrat);
    }
    printf("%s\n", all_ok ? "test_gradcheck: OK" : "test_gradcheck: FAIL");
    return all_ok ? 0 : 1;
}
