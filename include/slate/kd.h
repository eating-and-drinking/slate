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

#ifdef __cplusplus
}
#endif

#endif
