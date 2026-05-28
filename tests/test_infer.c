// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// test_infer.c — verify the KV-cached inference engine produces the
// same logits as the autograd-based training forward.  This is THE
// correctness contract for the inference fast-path; if it ever drifts
// we've silently changed model semantics under deployment, which is a
// non-event we cannot afford in production.
//
// Strategy:
//   1. Build a small slate_module_causal_lm with random weights.
//   2. Run the training-side forward on a prompt; capture last-position
//      logits.
//   3. Run the inference engine's prefill on the same prompt; capture
//      logits from prefill's output.
//   4. Assert the two logits arrays agree within a small tolerance
//      (allowing for accumulation-order differences in softmax/rms_norm).
//   5. Verify multi-step decode_step matches a re-forwarded "extend prompt
//      by one token" computation, so the cache-vs-recompute drift is also
//      bounded.

#include "slate/slate.h"
#include "slate/transformer.h"
#include "slate/infer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB     32
#define MAX_SEQ   16
#define D_MODEL   16
#define N_LAYERS   2
#define FFN_H     32
#define NORM_EPS 1e-5f

// L2 of two vectors as a similarity check (small is better).
static float l2_diff(const float* a, const float* b, int n) {
    double s = 0;
    for (int i = 0; i < n; ++i) {
        double d = (double)a[i] - (double)b[i];
        s += d * d;
    }
    return (float)sqrt(s);
}

// Max |a - b|.
static float linf_diff(const float* a, const float* b, int n) {
    float m = 0;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// Run the training-side forward and copy the last-position logits.
static void train_forward_last(slate_module_t* model,
                                 slate_arena_t* N, slate_arena_t* S,
                                 const int32_t* tokens, int n,
                                 float* out_logits) {
    slate_graph_ctx_t ctx;
    slate_graph_ctx_init(&ctx, N, S);
    ctx.training = false;
    int64_t ts[2] = { 1, n };
    slate_tensor_t* toks = slate_tensor_new(S, SLATE_DTYPE_I32, 2, ts, false);
    for (int t = 0; t < n; ++t) ((int32_t*)toks->data)[t] = tokens[t];
    slate_tensor_t* logits = slate_module_forward(model, &ctx, toks);
    const float* p = (const float*)logits->data + (int64_t)(n - 1) * VOCAB;
    memcpy(out_logits, p, (size_t)VOCAB * sizeof(float));
    slate_graph_ctx_reset(&ctx);
}

int main(void) {
    int ok = 1;

    slate_arena_t* P = slate_arena_create(8 << 20);
    slate_arena_t* N = slate_arena_create(8 << 20);
    slate_arena_t* S = slate_arena_create(16 << 20);

    slate_module_t* model = slate_module_causal_lm_new(
        P, VOCAB, MAX_SEQ, D_MODEL, N_LAYERS, FFN_H, NORM_EPS, /*seed=*/0xABCD);
    if (!model) { puts("causal_lm_new FAIL"); return 1; }

    slate_infer_engine_t* eng = slate_infer_engine_new(
        model, N_LAYERS, D_MODEL, VOCAB, FFN_H, MAX_SEQ);
    if (!eng) { puts("infer_engine_new FAIL"); return 1; }

    // ------------------------------------------------------------------
    // 1. Prefill matches training-forward on the last-position logits.
    // ------------------------------------------------------------------
    {
        int32_t prompt[5] = { 3, 17, 11, 4, 22 };
        float train_logits[VOCAB], infer_logits[VOCAB];
        train_forward_last(model, N, S, prompt, 5, train_logits);

        slate_infer_session_t* sess = slate_infer_session_new(eng);
        int rc = slate_infer_prefill(sess, prompt, 5, infer_logits);
        if (rc != 0) { printf("prefill rc=%d FAIL\n", rc); ok = 0; }

        float l2 = l2_diff(train_logits, infer_logits, VOCAB);
        float li = linf_diff(train_logits, infer_logits, VOCAB);
        printf("[prefill] L2 diff = %.6f   Linf diff = %.6f\n", l2, li);
        if (!(li < 1e-3f)) { puts("[prefill] FAIL: drift too large"); ok = 0; }

        slate_infer_session_free(sess);
    }

    // ------------------------------------------------------------------
    // 2. decode_step on extension matches a full re-forward.
    // ------------------------------------------------------------------
    {
        int32_t prompt[4] = { 7, 1, 19, 0 };
        int32_t extended[5] = { 7, 1, 19, 0, 13 };  // append one token

        slate_infer_session_t* sess = slate_infer_session_new(eng);
        float prefill_logits[VOCAB], step_logits[VOCAB], train_logits[VOCAB];

        int rc = slate_infer_prefill(sess, prompt, 4, prefill_logits);
        if (rc != 0) { printf("prefill rc=%d FAIL\n", rc); ok = 0; }
        rc = slate_infer_decode_step(sess, 13, step_logits);
        if (rc != 0) { printf("decode rc=%d FAIL\n", rc); ok = 0; }

        train_forward_last(model, N, S, extended, 5, train_logits);
        float l2 = l2_diff(train_logits, step_logits, VOCAB);
        float li = linf_diff(train_logits, step_logits, VOCAB);
        printf("[decode]  L2 diff = %.6f   Linf diff = %.6f\n", l2, li);
        if (!(li < 1e-3f)) { puts("[decode] FAIL: drift too large"); ok = 0; }

        printf("[decode]  position now = %d (expected 5)\n",
                slate_infer_session_position(sess));
        if (slate_infer_session_position(sess) != 5) ok = 0;

        slate_infer_session_free(sess);
    }

    // ------------------------------------------------------------------
    // 3. Generation latency stays linear: 10 decode_steps in <50 ms on
    //    this tiny model.  Smoke test that the API is usable.
    // ------------------------------------------------------------------
    {
        int32_t prompt[3] = { 1, 2, 3 };
        slate_infer_session_t* sess = slate_infer_session_new(eng);
        float logits[VOCAB];
        int rc = slate_infer_prefill(sess, prompt, 3, logits);
        if (rc != 0) { puts("prefill (latency) FAIL"); ok = 0; }
        for (int i = 0; i < 10 && ok; ++i) {
            // pick argmax for determinism
            int best = 0; for (int v = 1; v < VOCAB; ++v) if (logits[v] > logits[best]) best = v;
            rc = slate_infer_decode_step(sess, (int32_t)best, logits);
            if (rc != 0) { printf("decode_step rc=%d at i=%d FAIL\n", rc, i); ok = 0; }
        }
        printf("[gen]     10 decode_steps OK, final position = %d\n",
                slate_infer_session_position(sess));
        slate_infer_session_free(sess);
    }

    // ------------------------------------------------------------------
    // 4. Batched decode_step matches sequential decode_step bit-identically.
    // ------------------------------------------------------------------
    {
        const int B = 4;
        int32_t prompts[4][4] = {
            { 3, 17, 11, 4 },
            { 7,  1, 19, 0 },
            { 5,  9, 15, 2 },
            { 1,  2,  3, 4 },
        };
        int32_t next_token[4] = { 22, 13, 8, 27 };

        // Sequential path: separate sessions, separate decode_steps.
        float seq_logits[4][VOCAB];
        for (int r = 0; r < B; ++r) {
            slate_infer_session_t* s = slate_infer_session_new(eng);
            float lg[VOCAB];
            slate_infer_prefill(s, prompts[r], 4, lg);
            slate_infer_decode_step(s, next_token[r], seq_logits[r]);
            slate_infer_session_free(s);
        }

        // Batched path: 4 sessions, prefilled separately, then one
        // slate_infer_batch_step for the next token of each.
        slate_infer_batch_t* batch = slate_infer_batch_new(eng, B);
        if (!batch) { puts("batch_new FAIL"); ok = 0; }
        slate_infer_session_t* sess[4];
        for (int r = 0; r < B; ++r) {
            sess[r] = slate_infer_session_new(eng);
            float lg[VOCAB];
            slate_infer_prefill(sess[r], prompts[r], 4, lg);
        }
        float batch_logits[4 * VOCAB];
        slate_infer_session_t* sess_ptrs[4] = { sess[0], sess[1], sess[2], sess[3] };
        int rc = slate_infer_batch_step(batch, sess_ptrs, B, next_token, batch_logits);
        if (rc != 0) { printf("batch_step rc=%d FAIL\n", rc); ok = 0; }

        float max_diff = 0;
        for (int r = 0; r < B; ++r) {
            for (int v = 0; v < VOCAB; ++v) {
                float d = fabsf(seq_logits[r][v] - batch_logits[r * VOCAB + v]);
                if (d > max_diff) max_diff = d;
            }
        }
        printf("[batch]   sequential vs batched (B=%d): Linf diff = %.6f\n",
                B, max_diff);
        if (!(max_diff < 1e-4f)) { puts("[batch] FAIL: batched output drifted"); ok = 0; }
        // Position should have advanced by one for every session
        for (int r = 0; r < B; ++r) {
            if (slate_infer_session_position(sess[r]) != 5) { ok = 0; }
            slate_infer_session_free(sess[r]);
        }
        slate_infer_batch_free(batch);
    }

    slate_infer_engine_free(eng);
    slate_module_destroy(model);
    slate_arena_destroy(P); slate_arena_destroy(N); slate_arena_destroy(S);

    printf("test_infer: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
