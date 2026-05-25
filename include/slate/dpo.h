// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// dpo.h — Direct Preference Optimization loss.
//
//   loss = -log(σ(β * (log_p_chosen - log_p_rejected
//                       - log_p_chosen_ref + log_p_rejected_ref)))
//
// We compute log_p as the sum of token log-probabilities under the policy
// across the response tokens. Reference logits are precomputed (frozen base).

#ifndef SLATE_DPO_H
#define SLATE_DPO_H

#include "slate/types.h"
#include "slate/autograd.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compute DPO loss given:
//   chosen_logits   [B, T_c, V]  - student logits over chosen response tokens
//   chosen_targets  [B, T_c]     - chosen token ids (int32)
//   chosen_logps_ref[B]          - sum_t log p_ref(chosen_target_t)
//   rejected_logits [B, T_r, V]
//   rejected_targets[B, T_r]
//   rejected_logps_ref[B]
//   beta            scalar
// Returns scalar loss tensor.
//
// Note: ref log-probs are passed as precomputed scalars per sample
// (1D F32 tensor of length B). This keeps the differentiable graph small
// and matches how production trainers cache the ref policy's outputs.
slate_tensor_t* slate_op_dpo_loss(slate_graph_ctx_t* ctx,
                                   slate_tensor_t* chosen_logits,
                                   slate_tensor_t* chosen_targets,
                                   slate_tensor_t* chosen_logps_ref,
                                   slate_tensor_t* rejected_logits,
                                   slate_tensor_t* rejected_targets,
                                   slate_tensor_t* rejected_logps_ref,
                                   float beta);

// Helper: sum of per-token log-softmax(logits)[target] across the sequence.
// logits [B, T, V], targets [B, T] (i32), returns [B] (f32).
// Used to materialize chosen_logps_ref / rejected_logps_ref when we want them.
slate_tensor_t* slate_op_seq_logp(slate_graph_ctx_t* ctx,
                                   slate_tensor_t* logits,
                                   slate_tensor_t* targets);

#ifdef __cplusplus
}
#endif

#endif
