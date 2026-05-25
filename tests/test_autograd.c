// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// test_autograd — smoke test that a one-op graph runs forward + backward
// without exploding. The numerical correctness of every op is exercised by
// test_gradcheck.

#include "slate/slate.h"

#include <assert.h>
#include <stdio.h>

int main(void) {
    slate_arena_t* params = slate_arena_create(64 * 1024);
    slate_arena_t* nodes  = slate_arena_create(64 * 1024);
    slate_arena_t* scrat  = slate_arena_create(64 * 1024);

    slate_graph_ctx_t ctx;
    slate_graph_ctx_init(&ctx, nodes, scrat);

    // a [2x3] with requires_grad, b [3x2] with requires_grad.
    int64_t as[2] = {2, 3};
    int64_t bs[2] = {3, 2};
    slate_tensor_t* a = slate_tensor_new(params, SLATE_DTYPE_F32, 2, as, true);
    slate_tensor_t* b = slate_tensor_new(params, SLATE_DTYPE_F32, 2, bs, true);

    float av[6] = {1, 2, 3, 4, 5, 6};
    float bv[6] = {7, 8, 9, 10, 11, 12};
    slate_tensor_set_data(a, av, sizeof(av));
    slate_tensor_set_data(b, bv, sizeof(bv));

    slate_tensor_t* c = slate_op_matmul(&ctx, a, b);
    assert(c);
    assert(c->n_dims == 2 && c->shape[0] == 2 && c->shape[1] == 2);

    // Sum-reduce-by-MSE-against-zero to get a scalar for backward.
    int64_t zs[2] = {2, 2};
    slate_tensor_t* zero = slate_tensor_new(scrat, SLATE_DTYPE_F32, 2, zs, false);
    slate_tensor_zero(zero);
    slate_tensor_t* loss = slate_op_mse_loss(&ctx, c, zero);
    assert(loss);

    assert(slate_graph_backward(&ctx, loss) == SLATE_OK);
    // Gradients of a and b should be non-zero.
    int any_grad = 0;
    for (int i = 0; i < 6; ++i) if (((float*)a->grad)[i] != 0.f) { any_grad = 1; break; }
    assert(any_grad);

    slate_arena_destroy(params);
    slate_arena_destroy(nodes);
    slate_arena_destroy(scrat);
    printf("test_autograd: OK\n");
    return 0;
}
