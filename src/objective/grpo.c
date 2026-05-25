// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#include "slate/grpo.h"
#include "slate/dpo.h"
#include "slate/tensor.h"
#include "slate/autograd.h"
#include "slate/runtime.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

// loss = -(1/N) * sum_k A_k * sum_t log π_θ(y_k^t)
//      = -(1/N) * sum_k A_k * logp_seq_k
// where N = K (sequence-level) or N = sum_k T_k (token-level / DAPO).
//
// Backward through seq_logp (already differentiable). We wrap it with a tiny
// op that:
//   forward:  computes loss scalar
//   backward: d loss / d logp_seq[k] = -A_k / N

typedef struct {
    float* advantages;  // [K]
    float norm;         // N
    int64_t K;
} grpo_state_t;

static void grpo_backward(slate_graph_node_t* node) {
    slate_tensor_t* logp = node->inputs[0];
    slate_tensor_t* out = node->output;
    grpo_state_t* st = (grpo_state_t*)node->user_data;
    if (!logp->requires_grad || !logp->grad) return;
    float dy = ((const float*)out->grad)[0];
    float* dp = (float*)logp->grad;
    for (int64_t k = 0; k < st->K; ++k) {
        dp[k] += -st->advantages[k] / st->norm * dy;
    }
}

slate_tensor_t* slate_op_grpo_loss(slate_graph_ctx_t* ctx,
                                    slate_tensor_t* logits,
                                    slate_tensor_t* targets,
                                    slate_tensor_t* rewards,
                                    const slate_grpo_config_t* cfg) {
    if (!ctx || !logits || !targets || !rewards) return NULL;
    int K = (int)logits->shape[0];
    int T = (int)logits->shape[1];
    int default_drop_std = cfg ? cfg->drop_std_norm : 1;
    int default_token_loss = cfg ? cfg->token_level_loss : 1;
    int dynamic = cfg ? cfg->dynamic_sampling : 1;

    // Compute mean and (optionally) std of rewards.
    float* r = (float*)rewards->data;
    double rs = 0;
    for (int k = 0; k < K; ++k) rs += r[k];
    float mean = (float)(rs / (double)K);
    // Dynamic sampling: if all rewards equal, return loss=0 with no graph.
    int all_equal = 1;
    for (int k = 1; k < K; ++k) if (r[k] != r[0]) { all_equal = 0; break; }
    if (dynamic && all_equal) {
        int64_t s1[1] = {1};
        slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32, 1, s1, false);
        ((float*)out->data)[0] = 0.0f;
        if (!out->grad)
            out->grad = slate_arena_alloc(ctx->scratch_arena, sizeof(float), 16);
        return out;
    }
    float std_inv = 1.0f;
    if (!default_drop_std) {
        double vs = 0;
        for (int k = 0; k < K; ++k) vs += (r[k] - mean) * (r[k] - mean);
        float sd = sqrtf((float)(vs / (double)K) + 1e-8f);
        std_inv = 1.0f / sd;
    }
    float* adv = (float*)slate_arena_alloc(ctx->scratch_arena, (size_t)K * sizeof(float), 16);
    for (int k = 0; k < K; ++k) adv[k] = (r[k] - mean) * std_inv;

    // Get per-sequence log-prob via seq_logp.
    slate_tensor_t* logp = slate_op_seq_logp(ctx, logits, targets);

    // Norm: token-level = K*T, sequence-level = K.
    float norm = default_token_loss ? (float)(K * T) : (float)K;

    int64_t s1[1] = {1};
    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32, 1, s1, false);
    double L = 0;
    const float* pp = (const float*)logp->data;
    for (int k = 0; k < K; ++k) L += -(double)adv[k] * (double)pp[k];
    ((float*)out->data)[0] = (float)(L / (double)norm);

    slate_tensor_t* inputs[1] = {logp};
    slate_graph_node_t* node = slate_graph_record(ctx, "grpo_loss", inputs, 1, out, grpo_backward);
    if (node) {
        grpo_state_t* st = (grpo_state_t*)slate_arena_alloc(ctx->scratch_arena, sizeof(*st), 16);
        st->advantages = adv; st->norm = norm; st->K = K;
        node->user_data = st;
    }
    if (!out->grad) out->grad = slate_arena_alloc(ctx->scratch_arena, sizeof(float), 16);
    out->requires_grad = (node != NULL) || logits->requires_grad;
    return out;
}
