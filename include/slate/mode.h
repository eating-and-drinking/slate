// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// mode.h — day/night mode controller.
//
// Decides whether the machine is in INFERENCE ("day") or TRAINING ("night")
// state and supervises transitions based on idle, thermal, power, and time
// signals.
//
// IMPLEMENTATION STATUS: M5+.

#ifndef SLATE_MODE_H
#define SLATE_MODE_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum slate_machine_state {
    SLATE_STATE_INFERENCE   = 0,
    SLATE_STATE_TRAINING    = 1,
    SLATE_STATE_SUSPENDED   = 2,
    SLATE_STATE_TRANSITIONING = 3,
} slate_machine_state_t;

typedef struct slate_mode_policy {
    // Time windows when training is permitted (24-hour, local time).
    int  train_hour_start;
    int  train_hour_end;

    // Idle requirement (no input) before transitioning to training.
    int  idle_seconds_required;

    // Power requirements.
    bool require_ac_power;     // laptop
    bool require_charging;     // phone

    // Thermal thresholds, °C. -1 disables the check.
    int  thermal_soft_limit;   // throttle below this
    int  thermal_hard_limit;   // pause above this
    int  thermal_resume;       // resume below this

    // Storage requirements.
    int  min_free_gb;
} slate_mode_policy_t;

typedef struct slate_mode_controller slate_mode_controller_t;

slate_mode_controller_t* slate_mode_controller_new(const slate_mode_policy_t* policy);
void slate_mode_controller_destroy(slate_mode_controller_t* mc);

slate_machine_state_t slate_mode_controller_state(const slate_mode_controller_t* mc);

// Set/query the abort flag. Set by the controller when it wants the current
// training step to checkpoint and exit. Polled by the trainer every N steps.
bool slate_mode_controller_should_abort(const slate_mode_controller_t* mc);
void slate_mode_controller_request_abort(slate_mode_controller_t* mc);

// Get a recommended worker count based on current thermal state.
int slate_mode_controller_recommended_threads(const slate_mode_controller_t* mc,
                                               int max_threads);

#ifdef __cplusplus
}
#endif

#endif // SLATE_MODE_H
