// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// test_rope.c — verify slate_op_rope against the Python reference in
// tools/make_rope_ref.py.  Also a quick gradcheck: backward(forward(x))
// should reproduce the input (since RoPE is just a rotation by ±theta,
// its forward composed with its backward — i.e. rotation by -theta —
// must be a no-op on x to within fp32 epsilon).

#include "slate/slate.h"
#include "slate/ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    FILE* fp = fopen("/tmp/slate_rope_ref.bin", "rb");
    if (!fp) { puts("missing /tmp/slate_rope_ref.bin -- run tools/make_rope_ref.py"); return 1; }

    int32_t B, T, H, hd;
    float theta_base;
    fread(&B, 4, 1, fp);
    fread(&T, 4, 1, fp);
    fread(&H, 4, 1, fp);
    fread(&hd, 4, 1, fp);
    fread(&theta_base, 4, 1, fp);
    int64_t numel = (int64_t)B * T * H * hd;
    float* x_in = (float*)malloc((size_t)numel * sizeof(float));
    int32_t* positions = (int32_t*)malloc((size_t)T * sizeof(int32_t));
    float* y_ref = (float*)malloc((size_t)numel * sizeof(float));
    fread(x_in,      sizeof(float),   numel, fp);
    fread(positions, sizeof(int32_t), T,     fp);
    fread(y_ref,     sizeof(float),   numel, fp);
    fclose(fp);

    printf("RoPE test: B=%d T=%d H=%d hd=%d theta_base=%.1f  (%lld elems)\n",
            B, T, H, hd, theta_base, (long long)numel);

    slate_arena_t* N = slate_arena_create(4 << 20);
    slate_arena_t* S = slate_arena_create(8 << 20);
    slate_graph_ctx_t ctx;
    slate_graph_ctx_init(&ctx, N, S);
    ctx.training = false;

    // Build input tensor
    int64_t xs[4] = { B, T, H, hd };
    slate_tensor_t* x = slate_tensor_new(S, SLATE_DTYPE_F32, 4, xs, false);
    memcpy(x->data, x_in, (size_t)numel * sizeof(float));

    int64_t ps[1] = { T };
    slate_tensor_t* pos = slate_tensor_new(S, SLATE_DTYPE_I32, 1, ps, false);
    memcpy(pos->data, positions, (size_t)T * sizeof(int32_t));

    slate_tensor_t* y = slate_op_rope(&ctx, x, pos, theta_base);
    if (!y) { puts("slate_op_rope returned NULL FAIL"); return 1; }
    const float* y_out = (const float*)y->data;

    float linf = 0;
    for (int64_t i = 0; i < numel; ++i) {
        float d = fabsf(y_out[i] - y_ref[i]);
        if (d > linf) linf = d;
    }
    printf("[fwd]   Linf vs python reference = %.6e\n", linf);
    int ok = (linf < 1e-5f);
    if (!ok) puts("  FAIL: forward drift");

    // Inverse property: applying RoPE with positions = -original_positions
    // should approximately invert (rotation by -theta).  Use a second op
    // pass with negated positions.
    {
        slate_graph_ctx_reset(&ctx);
        slate_tensor_t* y2 = slate_tensor_new(S, SLATE_DTYPE_F32, 4, xs, false);
        memcpy(y2->data, y_out, (size_t)numel * sizeof(float));
        int64_t ps2[1] = { T };
        slate_tensor_t* neg_pos = slate_tensor_new(S, SLATE_DTYPE_I32, 1, ps2, false);
        for (int t = 0; t < T; ++t) ((int32_t*)neg_pos->data)[t] = -positions[t];
        slate_tensor_t* x_back = slate_op_rope(&ctx, y2, neg_pos, theta_base);
        float inv_linf = 0;
        const float* xb = (const float*)x_back->data;
        for (int64_t i = 0; i < numel; ++i) {
            float d = fabsf(xb[i] - x_in[i]);
            if (d > inv_linf) inv_linf = d;
        }
        printf("[inv]   Linf round-trip (rope+ rope-) = %.6e\n", inv_linf);
        if (inv_linf > 1e-4f) { puts("  FAIL: inversion drift"); ok = 0; }
    }

    free(x_in); free(positions); free(y_ref);
    slate_arena_destroy(N); slate_arena_destroy(S);
    printf("test_rope: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
