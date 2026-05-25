// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#ifndef SLATE_MODE_STATE_H
#define SLATE_MODE_STATE_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum slate_runtime_mode_value {
    SLATE_RM_INFERENCE = 0,
    SLATE_RM_TRAINING = 1,
    SLATE_RM_TEACHER_SCORING = 2,
} slate_runtime_mode_value_t;

typedef struct slate_runtime_state {
    slate_runtime_mode_value_t mode;
    int kv_cache_enabled;
    int grad_recording_enabled;
    int selective_checkpoint_mask;  // bitmask of which activations to keep
} slate_runtime_state_t;

void slate_runtime_set_mode(slate_runtime_state_t* s, slate_runtime_mode_value_t m);
const char* slate_runtime_mode_name(slate_runtime_mode_value_t m);

#ifdef __cplusplus
}
#endif

#endif
