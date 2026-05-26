// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#ifndef SLATE_KD_H
#define SLATE_KD_H

#include "slate/types.h"
#include "slate/autograd.h"

#ifdef __cplusplus
extern "C" {
#endif

// KD loss: KL(softmax(teacher / T) || softmax(student / T)) * T^2
// student_logits [B, T, V] : trainable
// teacher_logits [B, T, V] : pre-computed, treated as constants
// Returns scalar.
slate_tensor_t* slate_op_kd_loss(slate_graph_ctx_t* ctx,
                                  slate_tensor_t* student_logits,
                                  slate_tensor_t* teacher_logits,
                                  float temperature);

// Top-k KD loss: same KL as slate_op_kd_loss, but the teacher distribution
// is only specified on its top-k tokens per position. Useful for:
//   * remote teachers that only return top-k logits (HTTP, mmap'd cache);
//   * on-policy distillation with a large vocab where the full teacher
//     [B, T, V] tensor is wasteful.
//
// Semantics: the teacher distribution P~ is taken to be
//     softmax(topk_logits / T)   for the K tokens listed in topk_indices,
//     0                          everywhere else (i.e. truncated support).
// Student gradient: T * (Q_v - P~_v) / (B*T) per logit slot.
//
// student_logits [B, T, V] F32 : trainable
// topk_indices   [B, T, K] I32 : teacher's top-k vocab indices per position
// topk_logits    [B, T, K] F32 : teacher's logits at those indices (constants)
// Returns scalar.
slate_tensor_t* slate_op_kd_loss_topk(slate_graph_ctx_t* ctx,
                                       slate_tensor_t* student_logits,
                                       slate_tensor_t* topk_indices,
                                       slate_tensor_t* topk_logits,
                                       float temperature);

#ifdef __cplusplus
}
#endif

#endif
