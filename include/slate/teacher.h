// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// teacher.h — knowledge distillation teacher abstraction.
//
// Four implementations:
//   - SelfTeacher    : uses the frozen base, zero extra cost
//   - LocalTeacher   : runs a larger Slate model in-process or over TCP
//   - OpenAITeacher  : REST + top-k logprobs (white-box where k ≤ 20)
//   - AnthropicTeacher: REST (black-box only; can_score returns false)
//
// IMPLEMENTATION STATUS: M6.

#ifndef SLATE_TEACHER_H
#define SLATE_TEACHER_H

#include "slate/types.h"
#include "slate/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_teacher slate_teacher_t;

// A top-k logit slice for a single token position.
typedef struct slate_topk_slice {
    int32_t token_ids[64];  // up to k <= 64; remaining filled with -1
    float   logits[64];
    int     k;
} slate_topk_slice_t;

// Per-sequence top-k logits, [seq_len] entries.
typedef struct slate_topk_logits {
    slate_topk_slice_t* slices;
    int seq_len;
    int k;
} slate_topk_logits_t;

typedef struct slate_teacher_response {
    int32_t* tokens;
    int n_tokens;
    char* text;            // optional; NULL if teacher only returns tokens
} slate_teacher_response_t;

struct slate_teacher {
    bool (*can_score)(slate_teacher_t* self);
    int  (*max_k)(slate_teacher_t* self);

    slate_teacher_response_t (*generate)(slate_teacher_t* self,
                                         const int32_t* prompt_tokens,
                                         int n_prompt_tokens,
                                         int max_new_tokens);

    slate_topk_logits_t (*score)(slate_teacher_t* self,
                                 const int32_t* full_sequence,
                                 int seq_len,
                                 int k);

    void (*destroy)(slate_teacher_t* self);
    void* user_data;
};

// =============================================================================
// Concrete teacher constructors.
// =============================================================================

// SelfTeacher: uses the frozen base of the student. Always available.
slate_teacher_t* slate_teacher_self_new(slate_module_t* base);

// LocalTeacher (in-process): runs another loaded model.
slate_teacher_t* slate_teacher_local_new(slate_module_t* model);

// LocalTeacher (RPC): connects to a slate-serve process over TCP.
slate_teacher_t* slate_teacher_local_rpc_new(const char* host, int port);

// OpenAITeacher: requires OPENAI_API_KEY in environment.
slate_teacher_t* slate_teacher_openai_new(const char* model_name);

// AnthropicTeacher: requires ANTHROPIC_API_KEY. can_score() returns false.
slate_teacher_t* slate_teacher_anthropic_new(const char* model_name);

#ifdef __cplusplus
}
#endif

#endif // SLATE_TEACHER_H
