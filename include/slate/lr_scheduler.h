// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// lr_scheduler.h — learning-rate schedules.
//
// Schedulers are pure functions of step; no state required beyond the
// configuration. The trainer calls slate_lr_scheduler_get() once per step
// and pushes the result into the optimizer via slate_optimizer_set_lr().

#ifndef SLATE_LR_SCHEDULER_H
#define SLATE_LR_SCHEDULER_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_lr_scheduler slate_lr_scheduler_t;

// Constant LR (no scheduling).
slate_lr_scheduler_t* slate_lr_constant_new(float lr);

// Linear warmup for `warmup_steps`, then cosine decay from lr_max to lr_min
// over the remaining `total_steps - warmup_steps`.
slate_lr_scheduler_t* slate_lr_cosine_warmup_new(float lr_max,
                                                  float lr_min,
                                                  int warmup_steps,
                                                  int total_steps);

// Returns the LR at training step `step` (0-indexed).
float slate_lr_scheduler_get(const slate_lr_scheduler_t* s, int step);

void slate_lr_scheduler_destroy(slate_lr_scheduler_t* s);

// =============================================================================
// Gradient clipping.
// =============================================================================

// Global-norm gradient clipping. If ||grads||_2 > max_norm, scale all grads
// by max_norm / ||grads||_2. Returns the pre-clip global norm (useful for
// diagnostics).
float slate_clip_grad_norm(slate_param_set_t* ps, float max_norm);

#ifdef __cplusplus
}
#endif

#endif // SLATE_LR_SCHEDULER_H
