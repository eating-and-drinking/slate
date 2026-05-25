// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#include "slate/mode_state.h"
#include <stddef.h>

void slate_runtime_set_mode(slate_runtime_state_t* s, slate_runtime_mode_value_t m) {
    if (!s) return;
    s->mode = m;
    switch (m) {
        case SLATE_RM_INFERENCE:
            s->kv_cache_enabled = 1;
            s->grad_recording_enabled = 0;
            s->selective_checkpoint_mask = 0;
            break;
        case SLATE_RM_TRAINING:
            s->kv_cache_enabled = 0;
            s->grad_recording_enabled = 1;
            s->selective_checkpoint_mask = 1;  // block-input checkpointing on
            break;
        case SLATE_RM_TEACHER_SCORING:
            s->kv_cache_enabled = 0;
            s->grad_recording_enabled = 0;
            s->selective_checkpoint_mask = 0;
            break;
    }
}

const char* slate_runtime_mode_name(slate_runtime_mode_value_t m) {
    switch (m) {
        case SLATE_RM_INFERENCE:       return "INFERENCE";
        case SLATE_RM_TRAINING:        return "TRAINING";
        case SLATE_RM_TEACHER_SCORING: return "TEACHER_SCORING";
        default: return "?";
    }
}
