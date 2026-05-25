// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// kto.h — Kahneman-Tversky Optimization. Binary-feedback RLHF.

#ifndef SLATE_KTO_H
#define SLATE_KTO_H

#include "slate/types.h"
#include "slate/autograd.h"

#ifdef __cplusplus
extern "C" {
#endif

// KTO loss. For each example (logits, target, label):
//   r_θ = β * (logp_policy - logp_ref)
//   If label == +1 (desirable):    L = σ(KL_avg - r_θ)
//   If label == -1 (undesirable):  L = σ(r_θ - KL_avg)
// We use a simplified KL_avg = 0 (omit the KL term; original KTO uses an
// estimated batch-mean KL but for a smoke test the simpler form behaves the
// same in direction).
//
// `labels` is I32 of shape [B] with values +1 (good) / -1 (bad).
slate_tensor_t* slate_op_kto_loss(slate_graph_ctx_t* ctx,
                                   slate_tensor_t* logits,        // [B, T, V]
                                   slate_tensor_t* targets,       // [B, T] I32
                                   slate_tensor_t* logps_ref,     // [B] F32
                                   slate_tensor_t* labels,        // [B] I32 (+1/-1)
                                   float beta,
                                   float desirable_weight,
                                   float undesirable_weight);

#ifdef __cplusplus
}
#endif

#endif
