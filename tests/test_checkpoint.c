// SPDX-License-Identifier: Apache-2.0
// Verify: forward+backward through slate_op_checkpoint produces the SAME
// input gradient as forward+backward through the function directly. This
// proves rematerialization works.
//
// The function we wrap is a 3-step compute: y = silu(matmul(x, W) + b).
// We train against a fixed target. Then we compare:
//   (a) input gradient when computed via direct ops
//   (b) input gradient when wrapped in slate_op_checkpoint
// They must match to within floating-point noise.

#include "slate/slate.h"
#include "slate/checkpoint.h"
#include "slate/ops.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

typedef struct { slate_tensor_t* W; slate_tensor_t* b; } block_state_t;

static slate_tensor_t* block_fn(slate_graph_ctx_t* ctx, slate_tensor_t* x, void* ud) {
    block_state_t* s = (block_state_t*)ud;
    slate_tensor_t* y = slate_op_matmul(ctx, x, s->W);
    y = slate_op_add_bias(ctx, y, s->b);
    y = slate_op_silu(ctx, y);
    return y;
}

#define BS 4
#define D 6

int main(void) {
    slate_arena_t* P = slate_arena_create(2*1024*1024);
    slate_arena_t* N = slate_arena_create(4*1024*1024);
    slate_arena_t* S = slate_arena_create(8*1024*1024);

    int64_t ws[2] = {D, D}; int64_t bs[1] = {D};
    slate_tensor_t* W = slate_tensor_new(P, SLATE_DTYPE_F32, 2, ws, false);
    slate_tensor_t* b = slate_tensor_new(P, SLATE_DTYPE_F32, 1, bs, false);
    for (int i = 0; i < D*D; ++i) ((float*)W->data)[i] = 0.1f * (i + 1) - 1.5f;
    for (int i = 0; i < D; ++i)   ((float*)b->data)[i] = 0.05f * (i + 1);
    block_state_t bst = {W, b};

    int64_t xs[2] = {BS, D};
    slate_tensor_t* x1 = slate_tensor_new(P, SLATE_DTYPE_F32, 2, xs, true);
    slate_tensor_t* x2 = slate_tensor_new(P, SLATE_DTYPE_F32, 2, xs, true);
    for (int i = 0; i < BS*D; ++i) {
        float v = 0.1f * (i + 1) - 0.5f;
        ((float*)x1->data)[i] = v;
        ((float*)x2->data)[i] = v;  // identical
    }
    slate_tensor_t* tgt = slate_tensor_new(P, SLATE_DTYPE_F32, 2, xs, false);
    for (int i = 0; i < BS*D; ++i) ((float*)tgt->data)[i] = 0.07f * (i + 1);

    // Path 1: direct.
    slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, N, S); ctx.training = true;
    slate_tensor_zero_grad(x1);
    slate_tensor_t* y1 = block_fn(&ctx, x1, &bst);
    slate_tensor_t* L1 = slate_op_mse_loss(&ctx, y1, tgt);
    float loss1 = ((float*)L1->data)[0];
    slate_graph_backward(&ctx, L1);
    float g1[BS*D]; memcpy(g1, x1->grad, sizeof(g1));
    slate_graph_ctx_reset(&ctx);

    // Path 2: via checkpoint.
    slate_graph_ctx_init(&ctx, N, S); ctx.training = true;
    slate_tensor_zero_grad(x2);
    slate_tensor_t* y2 = slate_op_checkpoint(&ctx, x2, block_fn, &bst);
    slate_tensor_t* L2 = slate_op_mse_loss(&ctx, y2, tgt);
    float loss2 = ((float*)L2->data)[0];
    slate_graph_backward(&ctx, L2);
    float g2[BS*D]; memcpy(g2, x2->grad, sizeof(g2));
    slate_graph_ctx_reset(&ctx);

    // Compare
    printf("[ckpt] loss direct=%.6f  checkpoint=%.6f\n", loss1, loss2);
    float max_diff = 0;
    for (int i = 0; i < BS*D; ++i) {
        float d = fabsf(g1[i] - g2[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("[ckpt] max |grad_direct - grad_ckpt| = %.2e\n", max_diff);
    int ok = (fabsf(loss1 - loss2) < 1e-5f) && (max_diff < 1e-4f);
    // Show a couple of gradient values
    printf("[ckpt] direct grad[0..3]    = %.6f %.6f %.6f %.6f\n", g1[0], g1[1], g1[2], g1[3]);
    printf("[ckpt] checkpoint grad[0..3]= %.6f %.6f %.6f %.6f\n", g2[0], g2[1], g2[2], g2[3]);

    slate_arena_destroy(P); slate_arena_destroy(N); slate_arena_destroy(S);
    printf("test_checkpoint: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
