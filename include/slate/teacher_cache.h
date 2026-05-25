// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// teacher_cache.h — file-backed cache for teacher top-k logits.
//
// Format per record (binary, little-endian):
//   int32  prompt_id          : hash/id of the prompt
//   int32  seq_len
//   int32  k                  : top-k width
//   repeated seq_len * k tuples of (int32 token_id, float32 logit)

#ifndef SLATE_TEACHER_CACHE_H
#define SLATE_TEACHER_CACHE_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_teacher_cache slate_teacher_cache_t;

slate_teacher_cache_t* slate_teacher_cache_open(const char* path);
void slate_teacher_cache_close(slate_teacher_cache_t* c);

// Append a record. Atomic per-record write.
slate_status_t slate_teacher_cache_put(slate_teacher_cache_t* c,
                                        int32_t prompt_id, int seq_len, int k,
                                        const int32_t* topk_tokens,    // [seq_len*k]
                                        const float*   topk_logits);   // [seq_len*k]

// Look up a record. Returns SLATE_OK and fills outputs if found.
slate_status_t slate_teacher_cache_get(slate_teacher_cache_t* c,
                                        int32_t prompt_id,
                                        int* out_seq_len, int* out_k,
                                        int32_t** out_tokens, float** out_logits);

int slate_teacher_cache_size(slate_teacher_cache_t* c);

#ifdef __cplusplus
}
#endif

#endif
