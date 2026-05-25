// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// grpo.h — Group Relative Policy Optimization with Dr.GRPO + DAPO fixes.
//
// Algorithm sketch:
//   1. For one prompt, sample K responses from the policy (off this op).
//   2. Compute rewards r_1..r_K (off this op).
//   3. This op computes the loss given (logits_k, sampled_tokens_k, r_k):
//        Dr.GRPO advantage:   A_k = r_k - mean(r)           (no /std)
//        Loss (DAPO token-level):
//          L = -(1 / sum_k T_k) * sum_k sum_t A_k * log π_θ(y_k^t)
//   4. Backward flows to logits.
//
// We treat K as the batch dimension of `logits` (shape [K, T, V]). Rewards
// are passed as a [K] F32 tensor (constant, no gradient).

#ifndef SLATE_GRPO_H
#define SLATE_GRPO_H

#include "slate/types.h"
#include "slate/autograd.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_grpo_config {
    int   drop_std_norm;       // Dr.GRPO: 1 = drop /std normalization (default)
    int   token_level_loss;    // DAPO: 1 = average over total tokens not sequences
    int   dynamic_sampling;    // DAPO: 1 = if all rewards identical, return zero loss
} slate_grpo_config_t;

slate_tensor_t* slate_op_grpo_loss(slate_graph_ctx_t* ctx,
                                    slate_tensor_t* logits,    // [K, T, V] requires_grad
                                    slate_tensor_t* targets,   // [K, T] I32 sampled tokens
                                    slate_tensor_t* rewards,   // [K] F32 constants
                                    const slate_grpo_config_t* cfg);

#ifdef __cplusplus
}
#endif

#endif
