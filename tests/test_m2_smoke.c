#include "slate/slate.h"
#include "slate/ops.h"
#include <stdio.h>
int main(void) {
    slate_arena_t* a = slate_arena_create(4 * 1024 * 1024);
    slate_arena_t* n = slate_arena_create(4 * 1024 * 1024);
    slate_arena_t* s = slate_arena_create(4 * 1024 * 1024);
    slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, n, s);

    int64_t xs[2] = {2,3};
    slate_tensor_t* x = slate_tensor_new(a, SLATE_DTYPE_F32, 2, xs, true);
    for (int i = 0; i < 6; ++i) ((float*)x->data)[i] = 0.1f*(i+1);

    slate_tensor_t* y1 = slate_op_silu(&ctx, x); printf("silu OK shape [%lld,%lld]\n",(long long)y1->shape[0],(long long)y1->shape[1]);
    slate_tensor_t* y2 = slate_op_scale(&ctx, x, 2.0f); printf("scale OK first=%.3f\n",((float*)y2->data)[0]);
    slate_tensor_t* y3 = slate_op_transpose_last2(&ctx, x); printf("transpose OK shape [%lld,%lld]\n",(long long)y3->shape[0],(long long)y3->shape[1]);

    int64_t ws[1] = {3};
    slate_tensor_t* w = slate_tensor_new(a, SLATE_DTYPE_F32, 1, ws, true);
    for (int i = 0; i < 3; ++i) ((float*)w->data)[i] = 1.0f;
    slate_tensor_t* y4 = slate_op_rms_norm(&ctx, x, w, 1e-5f); printf("rms_norm OK first=%.3f\n",((float*)y4->data)[0]);

    int64_t es[2] = {5, 4};
    slate_tensor_t* W = slate_tensor_new(a, SLATE_DTYPE_F32, 2, es, true);
    for (int i = 0; i < 20; ++i) ((float*)W->data)[i] = (float)i * 0.1f;
    int64_t is[1] = {3};
    slate_tensor_t* idx = slate_tensor_new(a, SLATE_DTYPE_I32, 1, is, false);
    ((int32_t*)idx->data)[0]=0; ((int32_t*)idx->data)[1]=2; ((int32_t*)idx->data)[2]=4;
    slate_tensor_t* y5 = slate_op_embedding(&ctx, W, idx); printf("embedding OK shape [%lld,%lld]\n",(long long)y5->shape[0],(long long)y5->shape[1]);

    int64_t bs[3] = {2, 3, 4};
    int64_t bs2[3] = {2, 4, 3};
    slate_tensor_t* A = slate_tensor_new(a, SLATE_DTYPE_F32, 3, bs, true);
    slate_tensor_t* B = slate_tensor_new(a, SLATE_DTYPE_F32, 3, bs2, true);
    for (int i = 0; i < 24; ++i) { ((float*)A->data)[i] = 0.1f*i; ((float*)B->data)[i] = 0.05f*i; }
    slate_tensor_t* y6 = slate_op_bmm(&ctx, A, B); printf("bmm OK shape [%lld,%lld,%lld]\n",(long long)y6->shape[0],(long long)y6->shape[1],(long long)y6->shape[2]);

    int64_t ms[3] = {1, 3, 3};
    slate_tensor_t* M = slate_tensor_new(a, SLATE_DTYPE_F32, 3, ms, true);
    for (int i = 0; i < 9; ++i) ((float*)M->data)[i] = 1.0f;
    slate_tensor_t* y7 = slate_op_causal_mask(&ctx, M, 0.5f);
    printf("causal_mask OK first row: %.2f %.2f %.2f\n",
        ((float*)y7->data)[0], ((float*)y7->data)[1], ((float*)y7->data)[2]);
    printf("causal_mask    second row: %.2f %.2f %.2f\n",
        ((float*)y7->data)[3], ((float*)y7->data)[4], ((float*)y7->data)[5]);

    slate_arena_destroy(a); slate_arena_destroy(n); slate_arena_destroy(s);
    printf("all M2 ops: smoke test passed\n");
    return 0;
}
