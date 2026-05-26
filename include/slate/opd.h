// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// opd.h — On-Policy Distillation utilities.
//
// On-Policy Distillation (OPD) trains a student LM by:
//   1. sampling a rollout from the *student's current policy* (not from
//      a fixed dataset, and not from teacher rollouts);
//   2. scoring those student tokens with a frozen teacher;
//   3. minimising KL(teacher || student) on the student-generated tokens.
//
// The training signal is on-policy in the RL sense (data comes from
// π_θ) but the per-token loss is supervised KD — so OPD inherits the
// fast-convergence-per-FLOP of distillation while avoiding the
// off-policy distribution mismatch of vanilla KD.
//
// Slate doesn't bundle OPD into a single op because the training loop
// itself (sample → re-forward → KD) is short host code and is easier to
// audit, modify, and parallelise when written explicitly. This header
// only provides:
//
//   * slate_topk_extract — given a logits tensor [B, T, V], fill
//     [B, T, K] indices + logits with the per-position top-K. Used to
//     build the inputs to slate_op_kd_loss_topk from a teacher forward
//     pass.
//
// The expected OPD step (host code; see examples/07_opd/main.c for a
// runnable example) is:
//
//   // 1. Sample a rollout (no grad).
//   slate_graph_ctx_t sctx; slate_graph_ctx_init(&sctx, N, S);
//   sctx.training = false;
//   for (int g = 0; g < n_generate; ++g) {
//       slate_tensor_t* toks = ... fill with current sequence ...;
//       slate_tensor_t* lg  = slate_module_forward(student, &sctx, toks);
//       int next = slate_sample_token(/* last position of lg */, V, &cfg, rng);
//       seq[prefix_len + g] = next;
//       slate_graph_ctx_reset(&sctx);
//   }
//
//   // 2. Build the training graph on the full rollout.
//   slate_graph_ctx_t tctx; slate_graph_ctx_init(&tctx, N, S);
//   tctx.training = true;
//   slate_tensor_t* full_toks = ... fill with seq ...;
//   slate_tensor_t* s_logits = slate_module_forward(student, &tctx, full_toks);
//   slate_tensor_t* t_logits = slate_module_forward(teacher, &tctx, full_toks);
//   slate_tensor_t* idx_t  = ... [B, T, K] I32 ...;
//   slate_tensor_t* logs_t = ... [B, T, K] F32 ...;
//   slate_topk_extract((const float*)t_logits->data, B, T, V, K,
//                      (int32_t*)idx_t->data, (float*)logs_t->data);
//   slate_tensor_t* loss = slate_op_kd_loss_topk(&tctx, s_logits, idx_t,
//                                                logs_t, /*T=*/2.0f);
//
//   // 3. Step.
//   slate_optimizer_zero_grad(opt);
//   slate_graph_backward(&tctx, loss);
//   slate_optimizer_step(opt);
//   slate_graph_ctx_reset(&tctx);
//
// Pairs naturally with the Muon optimizer for the student matrix
// weights and AdamW (or SGD-momentum, which Muon falls back to) on
// non-matrix params.

#ifndef SLATE_OPD_H
#define SLATE_OPD_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Fill `out_indices[B*T*K]` and `out_logits[B*T*K]` with the per-position
// top-K of `logits[B*T*V]`. Uses a small O(V*K) selection (correct, but
// not the fastest for K close to V — fine for typical K=8..64).
//
// `out_indices` and `out_logits` must be pre-allocated with B*T*K slots.
// `logits` is interpreted row-major: position (b, t) starts at offset
// (b*T + t)*V.
void slate_topk_extract(const float* logits,
                         int B, int T, int V, int K,
                         int32_t* out_indices,
                         float*   out_logits);

#ifdef __cplusplus
}
#endif

#endif // SLATE_OPD_H
