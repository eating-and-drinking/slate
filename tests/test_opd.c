// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// test_opd.c — On-Policy Distillation toy verification.
//
// Setup: both teacher and student model "next-token logits" as an
// embedding lookup over [V, V]. Row i of the matrix gives the logits
// for the distribution of token_{t+1} given token_t = i. The teacher
// matrix is frozen with a known peaky pattern; the student starts
// uniform and must learn to match the teacher on tokens it visits.
//
// We check three things:
//   (1) slate_op_kd_loss_topk produces a finite, monotone-decreasing
//       loss when student logits are simply parameters (no rollout).
//       This is a sanity check on the new op's forward + backward.
//   (2) Full OPD loop: student generates rollouts itself; teacher
//       scores them; KD-topk loss is applied; loss decreases.
//   (3) After OPD training, on a fixed eval sequence the student's
//       argmax-next-token matches the teacher's on most positions.

#include "slate/slate.h"
#include "slate/kd.h"
#include "slate/opd.h"
#include "slate/sampling.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define V    8         // vocab size
#define K    3         // top-k for KD
#define T_F  2.0f      // KD temperature

// Make a "peaky" teacher: row i has its peak at (i+1) % V.
static void build_teacher(float* T_W) {
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) T_W[i*V + j] = 0.0f;
        T_W[i*V + (i + 1) % V] = 5.0f;
    }
}

// Tiny PRNG.
static uint64_t rng_state = 0xDEADBEEF;
static float frand01(uint64_t* s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((double)((*s >> 11) & ((1ULL << 53) - 1)) / (double)(1ULL << 53));
}

// ============================================================
// PART 1: sanity-check slate_op_kd_loss_topk with student logits
// as a direct parameter tensor (no module forward).
// ============================================================
static int part1_kd_topk_sanity(void) {
    slate_arena_t* P = slate_arena_create(2*1024*1024);
    slate_arena_t* O = slate_arena_create(2*1024*1024);
    slate_arena_t* N = slate_arena_create(2*1024*1024);
    slate_arena_t* S = slate_arena_create(4*1024*1024);

    int64_t ls[3] = {1, 4, V};
    slate_tensor_t* student = slate_tensor_new(P, SLATE_DTYPE_F32, 3, ls, true);
    for (int i = 0; i < 4*V; ++i) ((float*)student->data)[i] = 0.0f;  // uniform

    int64_t lk[3] = {1, 4, K};
    slate_tensor_t* idx = slate_tensor_new(P, SLATE_DTYPE_I32, 3, lk, false);
    slate_tensor_t* lg  = slate_tensor_new(P, SLATE_DTYPE_F32, 3, lk, false);
    // Per position, teacher's top-3 = {target, target+1, target-1} with logits {5, 1, 0}.
    int targets[4] = {3, 5, 1, 6};
    for (int t = 0; t < 4; ++t) {
        ((int32_t*)idx->data)[t*K + 0] = targets[t];
        ((int32_t*)idx->data)[t*K + 1] = (targets[t] + 1) % V;
        ((int32_t*)idx->data)[t*K + 2] = (targets[t] + V - 1) % V;
        ((float*)lg->data)[t*K + 0] = 5.0f;
        ((float*)lg->data)[t*K + 1] = 1.0f;
        ((float*)lg->data)[t*K + 2] = 0.0f;
    }

    slate_param_set_t ps; slate_param_set_init(&ps); slate_param_set_add(&ps, student);
    slate_optimizer_t* opt = slate_optimizer_adamw_new(O, &ps, 0.1f, 0.9f, 0.999f, 1e-8f, 0.0f);

    float L0 = 0, L1 = 0;
    for (int step = 0; step < 200; ++step) {
        slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, N, S); ctx.training = true;
        slate_tensor_t* loss = slate_op_kd_loss_topk(&ctx, student, idx, lg, T_F);
        float L = ((float*)loss->data)[0];
        if (step == 0) L0 = L;
        if (step == 199) L1 = L;
        slate_optimizer_zero_grad(opt);
        slate_graph_backward(&ctx, loss);
        slate_optimizer_step(opt);
        slate_graph_ctx_reset(&ctx);
    }
    int peaks_ok = 1;
    for (int t = 0; t < 4; ++t) {
        const float* row = (const float*)student->data + t*V;
        int peak = 0; for (int v = 1; v < V; ++v) if (row[v] > row[peak]) peak = v;
        if (peak != targets[t]) peaks_ok = 0;
    }
    printf("[part1] kd_loss_topk: loss %.4f -> %.4f   peaks_ok=%d\n", L0, L1, peaks_ok);

    slate_optimizer_destroy(opt); slate_param_set_destroy(&ps);
    slate_arena_destroy(P); slate_arena_destroy(O); slate_arena_destroy(N); slate_arena_destroy(S);
    return peaks_ok && (L1 < L0 * 0.3f);
}

// ============================================================
// PART 2 + 3: full OPD loop with a tiny "embedding-table" LM.
// ============================================================
static int part2_opd_loop(void) {
    slate_arena_t* P = slate_arena_create(2*1024*1024);
    slate_arena_t* O = slate_arena_create(2*1024*1024);
    slate_arena_t* N = slate_arena_create(4*1024*1024);
    slate_arena_t* S = slate_arena_create(8*1024*1024);

    // Teacher: frozen, peaky, "next = (i+1) % V".
    int64_t mw[2] = {V, V};
    slate_tensor_t* teacher_W = slate_tensor_new(P, SLATE_DTYPE_F32, 2, mw, false);
    build_teacher((float*)teacher_W->data);

    // Student: trainable, starts uniform.
    slate_tensor_t* student_W = slate_tensor_new(P, SLATE_DTYPE_F32, 2, mw, true);
    for (int i = 0; i < V*V; ++i) ((float*)student_W->data)[i] = 0.0f;

    slate_param_set_t ps; slate_param_set_init(&ps); slate_param_set_add(&ps, student_W);
    slate_optimizer_t* opt = slate_optimizer_adamw_new(O, &ps, 0.05f, 0.9f, 0.999f, 1e-8f, 0.0f);

    const int prefix_len = 1;
    const int n_generate = 7;
    const int seq_len    = prefix_len + n_generate;   // 8
    int32_t seq[16];

    float L_first = 0, L_last = 0;
    int n_steps = 250;
    for (int step = 0; step < n_steps; ++step) {
        // Choose a random prefix token.
        seq[0] = (int32_t)(((uint32_t)(frand01(&rng_state) * V)) % V);

        // --- 1. Sample rollout from student (no grad) ----------------------
        slate_sampler_config_t scfg = { .temperature = 1.0f, .top_k = 0,
                                         .top_p = 0.0f, .seed = 0 };
        for (int g = 0; g < n_generate; ++g) {
            int len_now = prefix_len + g;
            slate_graph_ctx_t sctx; slate_graph_ctx_init(&sctx, N, S); sctx.training = false;
            int64_t toks_shape[2] = { 1, len_now };
            slate_tensor_t* toks = slate_tensor_new(S, SLATE_DTYPE_I32, 2, toks_shape, false);
            for (int t = 0; t < len_now; ++t) ((int32_t*)toks->data)[t] = seq[t];
            slate_tensor_t* logits = slate_op_embedding(&sctx, student_W, toks);  // [1, len_now, V]
            const float* last = (const float*)logits->data + (len_now - 1) * V;
            int next = slate_sample_token(last, V, &scfg, &rng_state);
            seq[len_now] = (int32_t)next;
            slate_graph_ctx_reset(&sctx);
        }

        // --- 2. Build training graph on the full rollout (with grad) -------
        slate_graph_ctx_t tctx; slate_graph_ctx_init(&tctx, N, S); tctx.training = true;
        int64_t fs[2] = { 1, seq_len };
        slate_tensor_t* full_toks = slate_tensor_new(S, SLATE_DTYPE_I32, 2, fs, false);
        for (int t = 0; t < seq_len; ++t) ((int32_t*)full_toks->data)[t] = seq[t];

        slate_tensor_t* s_logits = slate_op_embedding(&tctx, student_W, full_toks);  // requires_grad
        slate_tensor_t* t_logits = slate_op_embedding(&tctx, teacher_W, full_toks);  // no grad

        // --- 3. Extract top-k from teacher --------------------------------
        int64_t lk[3] = { 1, seq_len, K };
        slate_tensor_t* idx_t  = slate_tensor_new(S, SLATE_DTYPE_I32, 3, lk, false);
        slate_tensor_t* logs_t = slate_tensor_new(S, SLATE_DTYPE_F32, 3, lk, false);
        slate_topk_extract((const float*)t_logits->data,
                            /*B=*/1, /*T=*/seq_len, /*V=*/V, /*K=*/K,
                            (int32_t*)idx_t->data, (float*)logs_t->data);

        // --- 4. KD top-k loss + step ---------------------------------------
        slate_tensor_t* loss = slate_op_kd_loss_topk(&tctx, s_logits, idx_t, logs_t, T_F);
        float L = ((float*)loss->data)[0];
        if (step == 0)         L_first = L;
        if (step == n_steps-1) L_last  = L;

        slate_optimizer_zero_grad(opt);
        slate_graph_backward(&tctx, loss);
        slate_optimizer_step(opt);
        slate_graph_ctx_reset(&tctx);
    }
    printf("[part2] OPD loop: loss %.4f -> %.4f   (drop %.1fx)\n",
            L_first, L_last,
            L_last > 1e-9f ? L_first / L_last : 0.0f);

    // --- Eval: argmax of student vs teacher on every input token. ----------
    int agree = 0;
    for (int i = 0; i < V; ++i) {
        const float* s_row = (const float*)student_W->data + i * V;
        const float* t_row = (const float*)teacher_W->data + i * V;
        int s_peak = 0, t_peak = 0;
        for (int v = 1; v < V; ++v) {
            if (s_row[v] > s_row[s_peak]) s_peak = v;
            if (t_row[v] > t_row[t_peak]) t_peak = v;
        }
        if (s_peak == t_peak) agree++;
    }
    printf("[part3] argmax agreement (student vs teacher): %d / %d\n", agree, V);

    int ok_loss   = (L_last < L_first * 0.5f);
    int ok_agree  = (agree >= V * 3 / 4);   // 75% of positions match

    slate_optimizer_destroy(opt); slate_param_set_destroy(&ps);
    slate_arena_destroy(P); slate_arena_destroy(O); slate_arena_destroy(N); slate_arena_destroy(S);
    return ok_loss && ok_agree;
}

int main(void) {
    int ok = 1;
    ok &= part1_kd_topk_sanity();
    ok &= part2_opd_loop();
    printf("test_opd: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
