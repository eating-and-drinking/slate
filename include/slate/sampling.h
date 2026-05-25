// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// sampling.h — token samplers for autoregressive generation.

#ifndef SLATE_SAMPLING_H
#define SLATE_SAMPLING_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_sampler_config {
    float temperature;   // 0 = greedy / argmax; >0 = sample
    int   top_k;         // 0 = disabled; >0 = restrict to top-k tokens
    float top_p;         // 0 = disabled; (0,1] = nucleus sampling
    uint64_t seed;
} slate_sampler_config_t;

// Sample one token from logits[vocab_size]. Returns the chosen token id.
// The RNG state is read+updated through `rng_state`.
int slate_sample_token(const float* logits, int vocab_size,
                        const slate_sampler_config_t* cfg,
                        uint64_t* rng_state);

#ifdef __cplusplus
}
#endif

#endif
