// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// bench_batch_throughput.c — measure tokens/sec as a function of
// continuous-batching width B.
//
// We run a small but production-shaped transformer (D=256, L=4,
// FFN=1024, V=4096) — about the smallest model where the GEMM kernel
// has enough work to actually amortise.  For each B ∈ {1, 2, 4, 8, 16}
// we open B sessions, prefill each with a 32-token prompt, then run
// 100 batched decode_steps and report:
//
//   * tokens/sec  — B * 100 / wall_time_sec
//   * speedup     — tokens/sec(B) / tokens/sec(1)
//
// Sequential baseline is the same total work (B*100 decode_steps) but
// each one as a separate decode_step call so the linear-projection
// GEMMs all run at M=1.

#include "slate/slate.h"
#include "slate/transformer.h"
#include "slate/infer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define VOCAB       4096
#define MAX_SEQ      256
#define D_MODEL      256
#define N_LAYERS       4
#define FFN_H       1024
#define PROMPT_LEN    32
#define GEN_STEPS    100

static double now_sec(void) {
    struct timeval t; gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec / 1e6;
}

static void run_for_batch(slate_infer_engine_t* eng, int B,
                           double* out_tps_seq, double* out_tps_batched) {
    // Prepare B sessions for the sequential path.
    slate_infer_session_t** sess_seq = (slate_infer_session_t**)
        calloc((size_t)B, sizeof(slate_infer_session_t*));
    slate_infer_session_t** sess_bat = (slate_infer_session_t**)
        calloc((size_t)B, sizeof(slate_infer_session_t*));
    for (int r = 0; r < B; ++r) {
        sess_seq[r] = slate_infer_session_new(eng);
        sess_bat[r] = slate_infer_session_new(eng);
    }

    int32_t prompt[PROMPT_LEN];
    for (int t = 0; t < PROMPT_LEN; ++t) prompt[t] = (t * 7 + 3) % VOCAB;
    float* logits = (float*)malloc(sizeof(float) * VOCAB);
    float* batch_logits = (float*)malloc(sizeof(float) * (size_t)B * VOCAB);

    for (int r = 0; r < B; ++r) {
        slate_infer_prefill(sess_seq[r], prompt, PROMPT_LEN, logits);
        slate_infer_prefill(sess_bat[r], prompt, PROMPT_LEN, logits);
    }

    // -- Sequential timing -----------------------------------------------
    int32_t* feed_tok = (int32_t*)calloc((size_t)B, sizeof(int32_t));
    for (int r = 0; r < B; ++r) feed_tok[r] = 1;

    double t0 = now_sec();
    for (int step = 0; step < GEN_STEPS; ++step) {
        for (int r = 0; r < B; ++r) {
            slate_infer_decode_step(sess_seq[r], feed_tok[r], logits);
            feed_tok[r] = (feed_tok[r] + 7) % VOCAB;
        }
    }
    double t1 = now_sec();
    *out_tps_seq = (double)(B * GEN_STEPS) / (t1 - t0);

    // -- Batched timing --------------------------------------------------
    slate_infer_batch_t* batch = slate_infer_batch_new(eng, B);
    for (int r = 0; r < B; ++r) feed_tok[r] = 1;

    t0 = now_sec();
    for (int step = 0; step < GEN_STEPS; ++step) {
        slate_infer_batch_step(batch, sess_bat, B, feed_tok, batch_logits);
        for (int r = 0; r < B; ++r) feed_tok[r] = (feed_tok[r] + 7) % VOCAB;
    }
    t1 = now_sec();
    *out_tps_batched = (double)(B * GEN_STEPS) / (t1 - t0);

    slate_infer_batch_free(batch);
    for (int r = 0; r < B; ++r) {
        slate_infer_session_free(sess_seq[r]);
        slate_infer_session_free(sess_bat[r]);
    }
    free(sess_seq); free(sess_bat);
    free(logits); free(batch_logits);
    free(feed_tok);
}

int main(void) {
    printf("=== slate continuous-batching throughput ===\n");
    printf("Model: V=%d D=%d L=%d FFN=%d   prompt=%d  steps=%d\n\n",
            VOCAB, D_MODEL, N_LAYERS, FFN_H, PROMPT_LEN, GEN_STEPS);

    slate_arena_t* P = slate_arena_create(64 << 20);
    slate_module_t* model = slate_module_causal_lm_new(
        P, VOCAB, MAX_SEQ, D_MODEL, N_LAYERS, FFN_H, 1e-5f, /*seed=*/42);
    if (!model) { puts("model build FAIL"); return 1; }

    slate_infer_engine_t* eng = slate_infer_engine_new(
        model, N_LAYERS, D_MODEL, VOCAB, FFN_H, MAX_SEQ);
    if (!eng) { puts("engine new FAIL"); return 1; }

    int Bs[] = { 1, 2, 4, 8, 16 };
    int nB = (int)(sizeof(Bs) / sizeof(Bs[0]));

    printf("%-6s  %-14s  %-14s  %-8s\n",
            "B", "seq tok/s", "batch tok/s", "speedup");
    printf("%-6s  %-14s  %-14s  %-8s\n",
            "-", "---------", "-----------", "-------");
    double tps_seq_1 = 0;
    for (int i = 0; i < nB; ++i) {
        double tps_seq, tps_bat;
        run_for_batch(eng, Bs[i], &tps_seq, &tps_bat);
        if (i == 0) tps_seq_1 = tps_seq;
        printf("%-6d  %-14.1f  %-14.1f  %-8.2fx\n",
                Bs[i], tps_seq, tps_bat,
                tps_seq_1 > 0 ? tps_bat / tps_seq_1 : 0.0);
    }

    slate_infer_engine_free(eng);
    slate_module_destroy(model);
    slate_arena_destroy(P);
    return 0;
}
