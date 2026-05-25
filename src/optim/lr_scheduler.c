// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// lr_scheduler.c — constant + cosine-warmup schedules, and global grad clip.

#include "slate/lr_scheduler.h"
#include "slate/module.h"
#include "slate/tensor.h"

#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef enum {
    LR_KIND_CONST,
    LR_KIND_COSINE_WARMUP,
} lr_kind_t;

struct slate_lr_scheduler {
    lr_kind_t kind;
    float lr_max, lr_min;
    int warmup_steps;
    int total_steps;
};

slate_lr_scheduler_t* slate_lr_constant_new(float lr) {
    slate_lr_scheduler_t* s = (slate_lr_scheduler_t*)calloc(1, sizeof(*s));
    s->kind = LR_KIND_CONST;
    s->lr_max = s->lr_min = lr;
    return s;
}

slate_lr_scheduler_t* slate_lr_cosine_warmup_new(float lr_max,
                                                  float lr_min,
                                                  int warmup_steps,
                                                  int total_steps) {
    slate_lr_scheduler_t* s = (slate_lr_scheduler_t*)calloc(1, sizeof(*s));
    s->kind = LR_KIND_COSINE_WARMUP;
    s->lr_max = lr_max;
    s->lr_min = lr_min;
    s->warmup_steps = warmup_steps;
    s->total_steps = total_steps;
    return s;
}

float slate_lr_scheduler_get(const slate_lr_scheduler_t* s, int step) {
    if (!s) return 0.0f;
    switch (s->kind) {
        case LR_KIND_CONST:
            return s->lr_max;
        case LR_KIND_COSINE_WARMUP: {
            if (step < s->warmup_steps) {
                if (s->warmup_steps <= 0) return s->lr_max;
                return s->lr_max * ((float)(step + 1) / (float)s->warmup_steps);
            }
            int decay_total = s->total_steps - s->warmup_steps;
            if (decay_total <= 0) return s->lr_min;
            int decay_step = step - s->warmup_steps;
            if (decay_step >= decay_total) return s->lr_min;
            float t = (float)decay_step / (float)decay_total;
            float cos_val = 0.5f * (1.0f + cosf((float)M_PI * t));
            return s->lr_min + (s->lr_max - s->lr_min) * cos_val;
        }
    }
    return 0.0f;
}

void slate_lr_scheduler_destroy(slate_lr_scheduler_t* s) {
    free(s);
}

// =============================================================================
// Gradient clipping
// =============================================================================
float slate_clip_grad_norm(slate_param_set_t* ps, float max_norm) {
    if (!ps || max_norm <= 0.0f) return 0.0f;

    double sumsq = 0.0;
    for (int i = 0; i < ps->n_params; ++i) {
        slate_tensor_t* t = ps->params[i];
        if (!t || !t->grad) continue;
        const float* g = (const float*)t->grad;
        int64_t n = slate_tensor_numel(t);
        for (int64_t j = 0; j < n; ++j) sumsq += (double)g[j] * (double)g[j];
    }
    float norm = (float)sqrt(sumsq);
    if (norm > max_norm) {
        float scale = max_norm / (norm + 1e-12f);
        for (int i = 0; i < ps->n_params; ++i) {
            slate_tensor_t* t = ps->params[i];
            if (!t || !t->grad) continue;
            float* g = (float*)t->grad;
            int64_t n = slate_tensor_numel(t);
            for (int64_t j = 0; j < n; ++j) g[j] *= scale;
        }
    }
    return norm;
}
