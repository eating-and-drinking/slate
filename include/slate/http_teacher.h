// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// http_teacher.h — interface for cloud-API teachers (OpenAI, Anthropic, ...).
//
// The Slate core ships interface stubs. Actual HTTP transport is the user's
// responsibility (link against libcurl, write a Python sidecar, etc.) and
// plugs in via slate_http_teacher_set_transport().

#ifndef SLATE_HTTP_TEACHER_H
#define SLATE_HTTP_TEACHER_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Result of a single API call.
typedef struct {
    char* text;             // generated text, malloc'd
    int32_t* token_ids;     // optional; provider-dependent. NULL if unavailable.
    float* top_logits;      // optional top-k logits when supported (OpenAI: yes,
                            //   Anthropic: no). NULL if unavailable.
    int n_tokens;
    int k;
    char* error;            // NULL on success
} slate_http_response_t;

void slate_http_response_free(slate_http_response_t* r);

// Transport callback: user supplies HTTP execution. Returns 0 on success.
typedef int (*slate_http_transport_fn)(const char* endpoint,
                                        const char* request_body,
                                        char** out_response_body,
                                        void* ud);

typedef struct slate_http_teacher slate_http_teacher_t;

slate_http_teacher_t* slate_http_teacher_openai_new(const char* model,
                                                     const char* api_key);
slate_http_teacher_t* slate_http_teacher_anthropic_new(const char* model,
                                                       const char* api_key);

// Plug a transport function. Until this is set, generate() returns an
// "transport not configured" error rather than making real network calls.
void slate_http_teacher_set_transport(slate_http_teacher_t* t,
                                       slate_http_transport_fn fn, void* ud);

// Generate / score (interface — OpenAI supports score with logprobs; Anthropic
// returns text only).
slate_http_response_t slate_http_teacher_generate(slate_http_teacher_t* t,
                                                   const char* prompt,
                                                   int max_tokens);

void slate_http_teacher_destroy(slate_http_teacher_t* t);

#ifdef __cplusplus
}
#endif

#endif
