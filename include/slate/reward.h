// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// reward.h — reward functions for RL objectives.
//
// A RewardFunction maps (prompt, response) -> scalar reward in some bounded
// range. Used by GRPO and its variants. For code RL, the typical reward is
// CompositeCodeReward(TestPass + Compile + Lint - LengthPenalty).
//
// IMPLEMENTATION STATUS: M7.

#ifndef SLATE_REWARD_H
#define SLATE_REWARD_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_reward slate_reward_t;
typedef struct slate_executor slate_executor_t;

struct slate_reward {
    const char* name;
    float reward_min;
    float reward_max;

    float (*score)(slate_reward_t* self,
                   const char* prompt,
                   const char* response);

    void (*destroy)(slate_reward_t* self);
    void* user_data;
};

// =============================================================================
// Built-in rewards.
// =============================================================================

// TestPassReward: runs response as code, returns pass rate over test cases.
// The executor handles sandboxing. The reward range is [-1, 1].
slate_reward_t* slate_reward_test_pass_new(slate_executor_t* executor,
                                            const char* test_runner_script);

// CompileReward: +0.5 if response parses and runs without error, 0 otherwise.
slate_reward_t* slate_reward_compile_new(slate_executor_t* executor);

// LinterReward: scaled style score in [0, 0.2].
slate_reward_t* slate_reward_linter_new(const char* linter_command);

// LengthPenalty: 0 if length within budget, negative beyond.
slate_reward_t* slate_reward_length_penalty_new(int soft_max, int hard_max);

// FormatReward: regex match on response.
slate_reward_t* slate_reward_format_new(const char* regex_pattern);

// CompositeReward: weighted sum of components.
typedef struct slate_reward_component {
    slate_reward_t* reward;
    float weight;
} slate_reward_component_t;

slate_reward_t* slate_reward_composite_new(const slate_reward_component_t* components,
                                            int n_components);

#ifdef __cplusplus
}
#endif

#endif // SLATE_REWARD_H
