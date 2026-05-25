// SPDX-License-Identifier: Apache-2.0
#include "slate/slate.h"
#include "slate/precision.h"
#include "slate/mode_state.h"
#include "slate/ops.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

slate_tensor_t* slate_op_matmul_bf16(slate_graph_ctx_t*, slate_tensor_t*, slate_tensor_t*);

#define M_ 4
#define K_ 8
#define N_ 4

int main(void) {
    int ok = 1;

    // ---- bf16 matmul vs fp32 baseline ----
    slate_arena_t* P = slate_arena_create(1*1024*1024);
    slate_arena_t* N = slate_arena_create(1*1024*1024);
    slate_arena_t* S = slate_arena_create(2*1024*1024);
    // Build fp32 reference
    float A_f32[M_*K_], B_f32[K_*N_];
    for (int i = 0; i < M_*K_; ++i) A_f32[i] = 0.1f * (i + 1);
    for (int i = 0; i < K_*N_; ++i) B_f32[i] = 0.05f * (i + 1) - 0.2f;
    float ref[M_*N_];
    memset(ref, 0, sizeof(ref));
    for (int i = 0; i < M_; ++i)
        for (int k = 0; k < K_; ++k)
            for (int j = 0; j < N_; ++j)
                ref[i*N_ + j] += A_f32[i*K_ + k] * B_f32[k*N_ + j];

    // Build bf16 tensors
    slate_tensor_t* A = slate_tensor_new(P, SLATE_DTYPE_BF16, 2, (int64_t[]){M_, K_}, false);
    slate_tensor_t* B = slate_tensor_new(P, SLATE_DTYPE_BF16, 2, (int64_t[]){K_, N_}, false);
    for (int i = 0; i < M_*K_; ++i) ((uint16_t*)A->data)[i] = slate_f32_to_bf16(A_f32[i]);
    for (int i = 0; i < K_*N_; ++i) ((uint16_t*)B->data)[i] = slate_f32_to_bf16(B_f32[i]);
    slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, N, S); ctx.training = false;
    slate_tensor_t* C = slate_op_matmul_bf16(&ctx, A, B);
    float max_err = 0;
    for (int i = 0; i < M_*N_; ++i) {
        float d = fabsf(((float*)C->data)[i] - ref[i]);
        if (d > max_err) max_err = d;
    }
    printf("[bf16] max abs error vs fp32 reference: %.4e\n", max_err);
    // bf16 has ~7 bit mantissa precision; expect relative ~1e-2.
    ok = ok && max_err < 0.05f;

    // ---- RuntimeMode state machine ----
    slate_runtime_state_t st;
    slate_runtime_set_mode(&st, SLATE_RM_INFERENCE);
    printf("[mode] INFERENCE: kv_cache=%d grad=%d ckpt=%d\n",
           st.kv_cache_enabled, st.grad_recording_enabled, st.selective_checkpoint_mask);
    ok = ok && st.kv_cache_enabled == 1 && st.grad_recording_enabled == 0;
    slate_runtime_set_mode(&st, SLATE_RM_TRAINING);
    printf("[mode] TRAINING:  kv_cache=%d grad=%d ckpt=%d\n",
           st.kv_cache_enabled, st.grad_recording_enabled, st.selective_checkpoint_mask);
    ok = ok && st.kv_cache_enabled == 0 && st.grad_recording_enabled == 1 && st.selective_checkpoint_mask;
    slate_runtime_set_mode(&st, SLATE_RM_TEACHER_SCORING);
    printf("[mode] TEACHER_SCORING: kv_cache=%d grad=%d ckpt=%d  name=%s\n",
           st.kv_cache_enabled, st.grad_recording_enabled, st.selective_checkpoint_mask,
           slate_runtime_mode_name(st.mode));
    ok = ok && st.kv_cache_enabled == 0 && st.grad_recording_enabled == 0;

    slate_arena_destroy(P); slate_arena_destroy(N); slate_arena_destroy(S);
    printf("test_bf16_mode: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
