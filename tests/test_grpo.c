// SPDX-License-Identifier: Apache-2.0
// Toy GRPO verification.
//
// Policy: single learnable [HID, V] linear over a constant input.
// Reward function: r(tokens) = (1/T) * (#tokens equal to "good_token") in [0,1].
// We expect the policy to learn to put all probability mass on good_token.

#include "slate/slate.h"
#include "slate/grpo.h"
#include "slate/ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define V 8
#define HID 8
#define K 6
#define T 3
#define STEPS 300

static int sample_from_logits(const float* logits, int vocab, uint64_t* rng) {
    float m = logits[0]; for (int i = 1; i < vocab; ++i) if (logits[i] > m) m = logits[i];
    double S = 0; float p[64];
    for (int i = 0; i < vocab; ++i) { p[i] = expf(logits[i] - m); S += p[i]; }
    float r = 0;
    *rng = *rng * 6364136223846793005ULL + 1442695040888963407ULL;
    r = (float)((double)((*rng >> 11) & ((1ULL<<24)-1)) / (1<<24));
    float c = 0;
    for (int i = 0; i < vocab; ++i) { c += (float)(p[i] / S); if (r <= c) return i; }
    return vocab - 1;
}

int main(void) {
    const int good = 7;
    slate_arena_t* P = slate_arena_create(2*1024*1024);
    slate_arena_t* O = slate_arena_create(2*1024*1024);
    slate_arena_t* N = slate_arena_create(4*1024*1024);
    slate_arena_t* S = slate_arena_create(8*1024*1024);
    int64_t ws[2] = {HID, V};
    slate_tensor_t* W = slate_tensor_new(P, SLATE_DTYPE_F32, 2, ws, true);
    uint64_t r = 0xC0DE;
    for (int i = 0; i < HID*V; ++i) {
        r = r * 6364136223846793005ULL + 1442695040888963407ULL;
        ((float*)W->data)[i] = (float)((double)((r >> 11) & ((1ULL<<24)-1)) / (1<<24) - 0.5) * 0.1f;
    }
    slate_param_set_t ps; slate_param_set_init(&ps); slate_param_set_add(&ps, W);
    slate_optimizer_t* opt = slate_optimizer_adamw_new(O, &ps, 0.05f, 0.9f, 0.999f, 1e-8f, 0.0f);

    float R0 = 0, Rn = 0;
    for (int step = 0; step < STEPS; ++step) {
        slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, N, S); ctx.training = true;
        // Input is a constant dense vector
        slate_tensor_t* x = slate_tensor_new(S, SLATE_DTYPE_F32, 2, (int64_t[]){1, HID}, false);
        for (int h = 0; h < HID; ++h) ((float*)x->data)[h] = 1.0f / HID;
        // Get logits [1, V]
        slate_tensor_t* logits_2d = slate_op_matmul(&ctx, x, W);

        // Sample K sequences of length T from the same distribution.
        // For autograd to flow we re-create a [K, T, V] logits tensor by tiling
        // the [V] vector — but we need autograd. Build [K, T, V] by tiling
        // a non-trainable copy isn't right (gradient won't flow). Solution:
        // use the SAME logits_2d for all K*T positions, via a manually-built
        // op? Easier: create logits as [K, T, V] directly from a [HID -> K*T*V]
        // matmul. But that changes the parameter shape.
        //
        // Simpler trick: replicate logits_2d into [K, T, V] by repeated matmul
        // with a fixed permutation? No.
        //
        // Cleanest: do the sample/reward bookkeeping OUTSIDE the graph,
        // then build a fresh graph that computes logp for those samples. The
        // GRPO loss only needs logits at the sampled positions; backward needs
        // logits to be differentiable. Approach: tile x to [K*T, HID] and
        // matmul to get [K*T, V], then view as [K, T, V].
        slate_tensor_t* x_tiled = slate_tensor_new(S, SLATE_DTYPE_F32, 2,
                                                    (int64_t[]){K*T, HID}, false);
        for (int i = 0; i < K*T; ++i)
            memcpy((float*)x_tiled->data + i * HID, x->data, HID * sizeof(float));
        slate_tensor_t* logits_flat = slate_op_matmul(&ctx, x_tiled, W);  // [K*T, V]
        slate_tensor_t* logits = slate_tensor_view(S, logits_flat, 3,
                                                    (int64_t[]){K, T, V}, NULL);

        // Step 1: sample K*T tokens off-graph from the SAME distribution as logits_flat[0].
        // For simplicity use the same logits for sampling each token.
        const float* pl = (const float*)logits_2d->data;
        slate_tensor_t* tgts = slate_tensor_new(S, SLATE_DTYPE_I32, 2, (int64_t[]){K, T}, false);
        slate_tensor_t* rew  = slate_tensor_new(S, SLATE_DTYPE_F32, 1, (int64_t[]){K}, false);
        for (int k = 0; k < K; ++k) {
            int hits = 0;
            for (int t = 0; t < T; ++t) {
                int tok = sample_from_logits(pl, V, &r);
                ((int32_t*)tgts->data)[k*T + t] = tok;
                if (tok == good) hits++;
            }
            ((float*)rew->data)[k] = (float)hits / (float)T;
        }
        // Mean reward over batch (for logging)
        float avg_R = 0;
        for (int k = 0; k < K; ++k) avg_R += ((float*)rew->data)[k];
        avg_R /= K;
        if (step == 0) R0 = avg_R;
        if (step == STEPS-1) Rn = avg_R;
        if (step % 50 == 0 || step == STEPS-1)
            printf("[grpo] step %3d  mean_reward=%.3f\n", step, avg_R);

        slate_grpo_config_t cfg = {1, 1, 1};   // Dr.GRPO advantage, DAPO token-level, dynamic sampling
        slate_tensor_t* loss = slate_op_grpo_loss(&ctx, logits, tgts, rew, &cfg);
        slate_optimizer_zero_grad(opt);
        slate_graph_backward(&ctx, loss);
        slate_clip_grad_norm(&ps, 1.0f);
        slate_optimizer_step(opt);
        slate_graph_ctx_reset(&ctx);
    }

    // Final inference: probability of good token
    slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, N, S); ctx.training = false;
    slate_tensor_t* x = slate_tensor_new(S, SLATE_DTYPE_F32, 2, (int64_t[]){1, HID}, false);
    for (int h = 0; h < HID; ++h) ((float*)x->data)[h] = 1.0f / HID;
    slate_tensor_t* logits = slate_op_matmul(&ctx, x, W);
    const float* pl = (const float*)logits->data;
    float m = pl[0]; for (int v = 1; v < V; ++v) if (pl[v] > m) m = pl[v];
    double Su = 0; float probs[V];
    for (int v = 0; v < V; ++v) { probs[v] = expf(pl[v] - m); Su += probs[v]; }
    for (int v = 0; v < V; ++v) probs[v] /= (float)Su;
    printf("[grpo] P(good=%d)=%.4f\n", good, probs[good]);
    printf("[grpo] mean reward %.3f -> %.3f (target: rising)\n", R0, Rn);
    int ok = (Rn > R0 * 2.0f) && (probs[good] > 0.5f);
    slate_graph_ctx_reset(&ctx);

    slate_optimizer_destroy(opt); slate_param_set_destroy(&ps);
    slate_arena_destroy(P); slate_arena_destroy(O); slate_arena_destroy(N); slate_arena_destroy(S);
    printf("test_grpo: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
