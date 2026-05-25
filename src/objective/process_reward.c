// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#define _POSIX_C_SOURCE 200809L
#include "slate/process_reward.h"
#include <float.h>
#include <stdlib.h>
#include <string.h>

float slate_process_reward(const char* response, const char* sep,
                            slate_prm_step_fn fn, void* ud,
                            slate_prm_agg_t agg) {
    if (!response || !sep || !fn) return 0.0f;
    char* buf = strdup(response);
    float total = (agg == SLATE_PRM_MIN) ? FLT_MAX : 0.0f;
    int n_steps = 0;
    char* save = NULL;
    char* tok = strtok_r(buf, sep, &save);
    while (tok) {
        float r = fn(tok, n_steps, ud);
        switch (agg) {
            case SLATE_PRM_SUM:  total += r; break;
            case SLATE_PRM_MEAN: total += r; break;
            case SLATE_PRM_MIN:  if (r < total) total = r; break;
        }
        n_steps++;
        tok = strtok_r(NULL, sep, &save);
    }
    free(buf);
    if (agg == SLATE_PRM_MEAN && n_steps > 0) total /= (float)n_steps;
    if (agg == SLATE_PRM_MIN && n_steps == 0) total = 0.0f;
    return total;
}
