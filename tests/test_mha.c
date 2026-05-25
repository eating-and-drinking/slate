#include "slate/slate.h"
#include "slate/transformer.h"
#include <stdio.h>
int main(void) {
    slate_arena_t* P = slate_arena_create(8*1024*1024);
    slate_arena_t* N = slate_arena_create(8*1024*1024);
    slate_arena_t* S = slate_arena_create(32*1024*1024);
    int B=2, T=8, D=32, H=4;
    slate_module_t* m = slate_module_mh_attention_new(P, D, H, 7);
    if (!m) { puts("FAIL alloc"); return 1; }
    slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, N, S); ctx.training = true;
    int64_t s[3]={B,T,D};
    slate_tensor_t* x = slate_tensor_new(P, SLATE_DTYPE_F32, 3, s, true);
    for (int i=0;i<B*T*D;++i) ((float*)x->data)[i] = 0.01f*(i+1);
    slate_tensor_t* y = slate_module_forward(m, &ctx, x);
    printf("mha out shape: [%lld,%lld,%lld]\n", (long long)y->shape[0],(long long)y->shape[1],(long long)y->shape[2]);
    // Backward: dummy loss
    slate_tensor_t* tgt = slate_tensor_new(P, SLATE_DTYPE_F32, 3, s, false);
    slate_tensor_t* L = slate_op_mse_loss(&ctx, y, tgt);
    slate_tensor_zero_grad(x);
    slate_graph_backward(&ctx, L);
    int nz = 0;
    for (int i=0;i<B*T*D;++i) if (((float*)x->grad)[i] != 0.0f) nz++;
    printf("mha backward: %d non-zero input grads (of %d)\n", nz, B*T*D);
    printf("test_mha: %s\n", nz > 0 ? "OK" : "FAIL");
    return nz > 0 ? 0 : 1;
}
