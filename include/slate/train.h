// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// train.h — high-level training loop.
//
// Ties together the autograd ctx, modules, optimizer, dataloader, objective,
// and (optionally) the streaming runtime and mode controller. The standard
// usage is one call to slate_trainer_run() per training run.
//
// IMPLEMENTATION STATUS: M1 (minimal), grows through M5+.

#ifndef SLATE_TRAIN_H
#define SLATE_TRAIN_H

#include "slate/types.h"
#include "slate/module.h"
#include "slate/optim.h"
#include "slate/objective.h"
#include "slate/data.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_trainer slate_trainer_t;

typedef struct slate_trainer_config {
    int max_steps;
    int log_every;
    int eval_every;
    int checkpoint_every;
    int grad_accum_steps;
    float grad_clip_norm;        // 0 disables
    const char* checkpoint_dir;
    uint64_t seed;
} slate_trainer_config_t;

slate_trainer_t* slate_trainer_new(slate_module_t* student,
                                    slate_optimizer_t* optimizer,
                                    slate_objective_t* objective,
                                    slate_dataloader_t* dataloader,
                                    const slate_trainer_config_t* cfg);

void slate_trainer_destroy(slate_trainer_t* tr);

slate_status_t slate_trainer_run(slate_trainer_t* tr);

// Resume from a checkpoint produced by a prior run.
slate_status_t slate_trainer_resume(slate_trainer_t* tr,
                                     const char* checkpoint_path);

#ifdef __cplusplus
}
#endif

#endif // SLATE_TRAIN_H
