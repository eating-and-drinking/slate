// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// 10_opd — On-Policy Distillation demo.
//
// Demonstrates the OPD pipeline:
//   (1) sample a rollout from the student's *current* policy;
//   (2) run the teacher on the same rollout to get reference logits;
//   (3) take the teacher's top-K per position;
//   (4) apply KL(teacher_topk || student) with slate_op_kd_loss_topk;
//   (5) backward + optimizer step.
//
// To keep the demo self-contained and finish in a few seconds with a
// clear convergence signal, both teacher and student are modelled as
// "next-token = embedding(token_id) over a [V, V] matrix": row i of
// the matrix gives the logits for the distribution of token_{t+1}
// given token_t = i.  This is the simplest non-trivial language model
// (an order-1 Markov chain) and exercises the *exact same* slate API
// surface (slate_op_embedding for forward, autograd, kd_loss_topk for
// loss) that a full slate_module_causal_lm would — so the OPD loop
// here is drop-in replaceable into a real transformer training script
// by swapping the forward call.
//
// The teacher's matrix is set to a peaky pattern where the most
// likely next token is (current + offset) % V — a structured signal
// the student can learn.  The student starts uniform.

#include "slate/slate.h"
#include "slate/kd.h"
#include "slate/opd.h"
#include "slate/sampling.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB        16
#define KD_K          4
#define KD_T        2.0f
#define PREFIX_LEN    1
#define GEN_LEN      11   // total rollout = PREFIX + GEN
#define N_STEPS     300
#define LR        0.05f

static uint64_t rng = 0xCAFEF00D;
static float frand01(uint64_t* s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((double)((*s >> 11) & ((1ULL << 53) - 1)) / (double)(1ULL << 53));
}

// Teacher: row i is peaked at (i + 3) % V with logit 5; runner-up at
// (i + 4) % V with logit 2; everything else near zero.
static void build_teacher(float* W) {
    for (int i = 0; i < VOCAB; ++i) {
        for (int j = 0; j < VOCAB; ++j) W[i*VOCAB + j] = 0.0f;
        W[i*VOCAB + (i + 3) % VOCAB] = 5.0f;
        W[i*VOCAB + (i + 4) % VOCAB] = 2.0f;
    }
}

// Single OPD step.  Returns rollout-side loss for logging.
static float opd_step(slate_tensor_t* student_W, slate_tensor_t* teacher_W,
                       slate_optimizer_t* opt,
                       slate_arena_t* N, slate_arena_t* S,
                       int32_t* seq) {
    const int seq_len = PREFIX_LEN + GEN_LEN;
    for (int t = 0; t < PREFIX_LEN; ++t) {
        seq[t] = (int32_t)(((uint32_t)(frand01(&rng) * VOCAB)) % VOCAB);
    }

    // --- 1. Sample rollout from student (no grad). -------------------------
    slate_sampler_config_t scfg = { .temperature = 1.0f, .top_k = 0,
                                     .top_p = 0.0f, .seed = 0 };
    for (int g = 0; g < GEN_LEN; ++g) {
        int len_now = PREFIX_LEN + g;
        slate_graph_ctx_t sctx; slate_graph_ctx_init(&sctx, N, S);
        sctx.training = false;
        int64_t ts[2] = { 1, len_now };
        slate_tensor_t* toks = slate_tensor_new(S, SLATE_DTYPE_I32, 2, ts, false);
        for (int t = 0; t < len_now; ++t) ((int32_t*)toks->data)[t] = seq[t];
        slate_tensor_t* logits = slate_op_embedding(&sctx, student_W, toks);
        const float* last = (const float*)logits->data + (int64_t)(len_now - 1) * VOCAB;
        seq[len_now] = (int32_t)slate_sample_token(last, VOCAB, &scfg, &rng);
        slate_graph_ctx_reset(&sctx);
    }

    // --- 2. Training forward on the full rollout (with grad). --------------
    slate_graph_ctx_t tctx; slate_graph_ctx_init(&tctx, N, S);
    tctx.training = true;
    int64_t fs[2] = { 1, seq_len };
    slate_tensor_t* full_toks = slate_tensor_new(S, SLATE_DTYPE_I32, 2, fs, false);
    for (int t = 0; t < seq_len; ++t) ((int32_t*)full_toks->data)[t] = seq[t];

    slate_tensor_t* s_logits = slate_op_embedding(&tctx, student_W, full_toks);
    slate_tensor_t* t_logits = slate_op_embedding(&tctx, teacher_W, full_toks);

    // --- 3. Teacher top-K. -------------------------------------------------
    int64_t lk[3] = { 1, seq_len, KD_K };
    slate_tensor_t* idx_t  = slate_tensor_new(S, SLATE_DTYPE_I32, 3, lk, false);
    slate_tensor_t* logs_t = slate_tensor_new(S, SLATE_DTYPE_F32, 3, lk, false);
    slate_topk_extract((const float*)t_logits->data,
                        1, seq_len, VOCAB, KD_K,
                        (int32_t*)idx_t->data, (float*)logs_t->data);

    // --- 4 + 5. Loss + backward + step. ------------------------------------
    slate_tensor_t* loss = slate_op_kd_loss_topk(&tctx, s_logits, idx_t, logs_t, KD_T);
    float L = ((float*)loss->data)[0];

    slate_optimizer_zero_grad(opt);
    slate_graph_backward(&tctx, loss);
    slate_optimizer_step(opt);
    slate_graph_ctx_reset(&tctx);
    return L;
}

// Fixed-eval metric: argmax agreement and per-position KL on every
// input token (which is also every position the student visits when
// rolling forward), so it reports global match quality not just match
// on whatever rollout the latest step happened to sample.
static void eval_fixed(slate_tensor_t* student_W, slate_tensor_t* teacher_W,
                        int* out_agree, float* out_kl) {
    const float* sw = (const float*)student_W->data;
    const float* tw = (const float*)teacher_W->data;
    int agree = 0;
    double kl_sum = 0;
    for (int i = 0; i < VOCAB; ++i) {
        const float* sr = sw + i * VOCAB;
        const float* tr = tw + i * VOCAB;
        int sp = 0, tp = 0;
        float sm = sr[0], tm = tr[0];
        for (int v = 1; v < VOCAB; ++v) {
            if (sr[v] > sm) sm = sr[v];
            if (tr[v] > tm) tm = tr[v];
            if (sr[v] > sr[sp]) sp = v;
            if (tr[v] > tr[tp]) tp = v;
        }
        if (sp == tp) agree++;
        double Ss = 0, St = 0;
        for (int v = 0; v < VOCAB; ++v) { Ss += exp(sr[v] - sm); St += exp(tr[v] - tm); }
        for (int v = 0; v < VOCAB; ++v) {
            double P = exp(tr[v] - tm) / St;
            double Q = exp(sr[v] - sm) / Ss;
            if (P > 1e-20) kl_sum += P * (log(P) - log(Q + 1e-20));
        }
    }
    *out_agree = agree;
    *out_kl    = (float)(kl_sum / VOCAB);
}

int main(void) {
    printf("=== Slate OPD demo ===\n");
    printf("teacher = next-token table over V=%d   K=%d top-k for KD\n", VOCAB, KD_K);

    slate_arena_t* P = slate_arena_create(2 << 20);
    slate_arena_t* O = slate_arena_create(4 << 20);
    slate_arena_t* N = slate_arena_create(8 << 20);
    slate_arena_t* S = slate_arena_create(16 << 20);

    // Teacher: frozen, peaky.
    int64_t mw[2] = { VOCAB, VOCAB };
    slate_tensor_t* teacher_W = slate_tensor_new(P, SLATE_DTYPE_F32, 2, mw, false);
    build_teacher((float*)teacher_W->data);

    // Student: trainable, uniform init.
    slate_tensor_t* student_W = slate_tensor_new(P, SLATE_DTYPE_F32, 2, mw, true);
    for (int i = 0; i < VOCAB*VOCAB; ++i) ((float*)student_W->data)[i] = 0.0f;

    slate_param_set_t ps; slate_param_set_init(&ps); slate_param_set_add(&ps, student_W);
    slate_optimizer_t* opt = slate_optimizer_adamw_new(O, &ps,
                                                       LR, 0.9f, 0.95f, 1e-8f, 0.0f);

    int32_t seq[64];
    int   agree0; float kl0;
    eval_fixed(student_W, teacher_W, &agree0, &kl0);
    printf("initial   eval-KL=%.4f   argmax-agree=%d/%d\n", kl0, agree0, VOCAB);

    for (int step = 0; step < N_STEPS; ++step) {
        float L = opd_step(student_W, teacher_W, opt, N, S, seq);
        if (step % 50 == 0 || step == N_STEPS - 1) {
            int a; float kl;
            eval_fixed(student_W, teacher_W, &a, &kl);
            printf("step %4d  rollout_loss=%.4f  eval-KL=%.4f  agree=%d/%d\n",
                    step, L, kl, a, VOCAB);
        }
    }
    int   agree1; float kl1;
    eval_fixed(student_W, teacher_W, &agree1, &kl1);

    printf("\n=== Summary ===\n");
    printf("Note: rollout-loss is on the student's *current* on-policy\n"
            "samples and is not directly monotone — eval-KL is the right\n"
            "convergence signal.\n");
    printf("eval-KL:          %.4f -> %.4f   (drop %.1fx)\n",
            kl0, kl1, kl1 > 1e-9f ? kl0 / kl1 : 0.0f);
    printf("argmax-agree:     %d/%d -> %d/%d\n", agree0, VOCAB, agree1, VOCAB);
    int ok = (kl1 < kl0 * 0.5f) && (agree1 > agree0);
    printf("OPD demo: %s\n", ok ? "OK" : "FAIL");

    slate_optimizer_destroy(opt);
    slate_param_set_destroy(&ps);
    slate_arena_destroy(P); slate_arena_destroy(O);
    slate_arena_destroy(N); slate_arena_destroy(S);
    return ok ? 0 : 1;
}
