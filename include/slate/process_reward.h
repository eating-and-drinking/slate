// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#ifndef SLATE_PROCESS_REWARD_H
#define SLATE_PROCESS_REWARD_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// A "process reward" treats a generated response as a sequence of steps and
// scores each step independently. Final aggregation can be sum, mean, or
// min (worst step dominates).
typedef enum {
    SLATE_PRM_SUM = 0,
    SLATE_PRM_MEAN = 1,
    SLATE_PRM_MIN = 2,
} slate_prm_agg_t;

// Score a step (rolling): given the step text, return a reward in [-1, 1].
typedef float (*slate_prm_step_fn)(const char* step_text, int step_idx, void* ud);

// Compute total reward for a response: split by `step_separator`, score each
// part with `step_fn`, aggregate with `agg`.
float slate_process_reward(const char* response,
                            const char* step_separator,
                            slate_prm_step_fn step_fn, void* ud,
                            slate_prm_agg_t agg);

#ifdef __cplusplus
}
#endif

#endif
