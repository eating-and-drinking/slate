// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// optim.h — L5: optimizers.
//
// An optimizer operates on a slate_param_set_t. It owns its optimizer state
// (momentum buffers, second-moment estimates, etc.) on an arena the caller
// provides.

#ifndef SLATE_OPTIM_H
#define SLATE_OPTIM_H

#include "slate/types.h"
#include "slate/module.h"

#ifdef __cplusplus
extern "C" {
#endif

struct slate_optimizer {
    void (*step)(slate_optimizer_t* self);
    void (*zero_grad)(slate_optimizer_t* self);
    void (*set_lr)(slate_optimizer_t* self, float lr);
    void (*destroy)(slate_optimizer_t* self);
    void* user_data;
};

static inline void slate_optimizer_set_lr(slate_optimizer_t* opt, float lr) {
    if (opt && opt->set_lr) opt->set_lr(opt, lr);
}

static inline void slate_optimizer_step(slate_optimizer_t* opt) {
    opt->step(opt);
}

static inline void slate_optimizer_zero_grad(slate_optimizer_t* opt) {
    opt->zero_grad(opt);
}

static inline void slate_optimizer_destroy(slate_optimizer_t* opt) {
    if (opt && opt->destroy) opt->destroy(opt);
}

// =============================================================================
// SGD with optional momentum.
// =============================================================================
//
// `params` must remain valid for the lifetime of the optimizer.
slate_optimizer_t* slate_optimizer_sgd_new(slate_arena_t* state_arena,
                                           slate_param_set_t* params,
                                           float learning_rate,
                                           float momentum);

// =============================================================================
// AdamW: Adam with decoupled weight decay.
// =============================================================================
//
// Standard transformer-default settings: lr=3e-4, betas=(0.9, 0.95),
// eps=1e-8, weight_decay=0.1.
slate_optimizer_t* slate_optimizer_adamw_new(slate_arena_t* state_arena,
                                              slate_param_set_t* params,
                                              float learning_rate,
                                              float beta1,
                                              float beta2,
                                              float eps,
                                              float weight_decay);

#ifdef __cplusplus
}
#endif

#endif // SLATE_OPTIM_H
