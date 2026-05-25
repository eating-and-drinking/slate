// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// objective.h — training objective abstraction.
//
// Objectives are pluggable loss functions: SFT, KD, DPO, KTO, GRPO with
// improvements, and Mixed (weighted combination). The training loop is
// objective-agnostic; it calls compute_loss() once per step.
//
// IMPLEMENTATION STATUS: M6-M7.

#ifndef SLATE_OBJECTIVE_H
#define SLATE_OBJECTIVE_H

#include "slate/types.h"
#include "slate/tensor.h"
#include "slate/module.h"
#include "slate/teacher.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_objective slate_objective_t;
typedef struct slate_reward slate_reward_t;
typedef struct slate_batch slate_batch_t;

// A batch may carry different fields depending on the objective. We use a
// tagged union pattern: the batch carries pointers that may be NULL, and the
// objective uses the ones it needs.
struct slate_batch {
    // Always present
    int32_t* input_tokens;
    int      seq_len;
    int      batch_size;

    // SFT / KD
    int32_t* target_tokens;   // hard labels
    float*   loss_mask;       // optional per-token weighting

    // KD (off-policy from teacher cache)
    slate_topk_logits_t* teacher_logits;

    // DPO / KTO
    int32_t* chosen_tokens;   int n_chosen;
    int32_t* rejected_tokens; int n_rejected;  // DPO only
    int*     binary_labels;   // KTO: +1 (good) / -1 (bad), per sample

    // GRPO (filled by trainer after sampling K responses)
    int32_t** sampled_tokens; // [K][seq_len]
    int       n_samples;
    float*    rewards;        // [K]
};

struct slate_objective {
    const char* name;
    bool (*needs_pair_data)   (slate_objective_t* self);
    bool (*needs_reference)   (slate_objective_t* self);
    bool (*needs_generation)  (slate_objective_t* self);
    int  (*n_samples_per_step)(slate_objective_t* self);

    slate_tensor_t* (*compute_loss)(slate_objective_t* self,
                                    slate_graph_ctx_t* ctx,
                                    const slate_batch_t* batch,
                                    slate_module_t* student,
                                    slate_module_t* reference,
                                    slate_teacher_t* teacher,
                                    slate_reward_t* reward);

    void (*destroy)(slate_objective_t* self);
    void* user_data;
};

// =============================================================================
// Constructors.
// =============================================================================

slate_objective_t* slate_objective_sft_new(void);

slate_objective_t* slate_objective_kd_new(float alpha,
                                           float temperature,
                                           bool on_policy);

typedef struct slate_dpo_config {
    float beta;          // 0.1 - 0.5, default 0.1
} slate_dpo_config_t;
slate_objective_t* slate_objective_dpo_new(const slate_dpo_config_t* cfg);

typedef struct slate_kto_config {
    float beta;          // typically 0.1
    float desirable_w;   // weight on good examples (default 1.0)
    float undesirable_w; // weight on bad examples (default 1.0)
} slate_kto_config_t;
slate_objective_t* slate_objective_kto_new(const slate_kto_config_t* cfg);

// GRPO + Dr.GRPO advantage + DAPO four-piece improvements.
typedef struct slate_dapo_drgrpo_config {
    int   K;                       // group size, default 4-8
    float temperature;             // sampling temperature, default 0.8
    int   max_response_tokens;
    float kl_beta;                 // KL penalty against ref, default 0.04
    float clip_low;                // default 0.2
    float clip_high;               // default 0.28 (asymmetric: clip-higher)
    bool  dynamic_sampling;        // skip degenerate groups, default true
    bool  token_level_loss;        // DAPO token-level, default true
    bool  drop_std_norm;           // Dr.GRPO: drop /std, default true
    bool  drop_length_norm;        // Dr.GRPO: drop /len, default true
    int   length_penalty_threshold;
} slate_dapo_drgrpo_config_t;
slate_objective_t* slate_objective_dapo_drgrpo_new(const slate_dapo_drgrpo_config_t* cfg);

// Mixed: weighted sum. Components are not owned; caller must keep them alive.
typedef struct slate_objective_component {
    slate_objective_t* obj;
    float weight;
} slate_objective_component_t;

slate_objective_t* slate_objective_mixed_new(const slate_objective_component_t* components,
                                              int n_components);

#ifdef __cplusplus
}
#endif

#endif // SLATE_OBJECTIVE_H
