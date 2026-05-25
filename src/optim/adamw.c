// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// adamw.c — Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

#include "slate/optim.h"
#include "slate/tensor.h"
#include "slate/module.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct adamw_state {
    slate_param_set_t* params;
    float lr;
    float beta1, beta2;
    float eps;
    float weight_decay;
    int   step_count;
    float** m;   // first moment
    float** v;   // second moment
} adamw_state_t;

static void adamw_step(slate_optimizer_t* self) {
    adamw_state_t* st = (adamw_state_t*)self->user_data;
    st->step_count++;
    float t = (float)st->step_count;
    float bc1 = 1.0f - powf(st->beta1, t);
    float bc2 = 1.0f - powf(st->beta2, t);

    for (int p = 0; p < st->params->n_params; ++p) {
        slate_tensor_t* tensor = st->params->params[p];
        if (!tensor || !tensor->grad) continue;
        float* w = (float*)tensor->data;
        const float* g = (const float*)tensor->grad;
        float* m = st->m[p];
        float* v = st->v[p];
        int64_t n = slate_tensor_numel(tensor);

        for (int64_t i = 0; i < n; ++i) {
            // AdamW: decoupled weight decay (not added to gradient).
            w[i] -= st->lr * st->weight_decay * w[i];

            m[i] = st->beta1 * m[i] + (1.0f - st->beta1) * g[i];
            v[i] = st->beta2 * v[i] + (1.0f - st->beta2) * g[i] * g[i];

            float m_hat = m[i] / bc1;
            float v_hat = v[i] / bc2;
            w[i] -= st->lr * m_hat / (sqrtf(v_hat) + st->eps);
        }
    }
}

static void adamw_zero_grad(slate_optimizer_t* self) {
    adamw_state_t* st = (adamw_state_t*)self->user_data;
    for (int p = 0; p < st->params->n_params; ++p) {
        slate_tensor_zero_grad(st->params->params[p]);
    }
}

static void adamw_set_lr(slate_optimizer_t* self, float lr) {
    adamw_state_t* st = (adamw_state_t*)self->user_data;
    st->lr = lr;
}

static void adamw_destroy(slate_optimizer_t* self) {
    if (!self) return;
    adamw_state_t* st = (adamw_state_t*)self->user_data;
    if (st) {
        free(st->m);
        free(st->v);
        free(st);
    }
    free(self);
}

slate_optimizer_t* slate_optimizer_adamw_new(slate_arena_t* state_arena,
                                              slate_param_set_t* params,
                                              float learning_rate,
                                              float beta1,
                                              float beta2,
                                              float eps,
                                              float weight_decay) {
    slate_optimizer_t* opt = (slate_optimizer_t*)calloc(1, sizeof(*opt));
    adamw_state_t* st = (adamw_state_t*)calloc(1, sizeof(*st));
    if (!opt || !st) { free(opt); free(st); return NULL; }

    st->params = params;
    st->lr = learning_rate;
    st->beta1 = beta1;
    st->beta2 = beta2;
    st->eps = eps;
    st->weight_decay = weight_decay;
    st->step_count = 0;

    st->m = (float**)calloc((size_t)params->n_params, sizeof(float*));
    st->v = (float**)calloc((size_t)params->n_params, sizeof(float*));
    for (int p = 0; p < params->n_params; ++p) {
        slate_tensor_t* t = params->params[p];
        int64_t n = slate_tensor_numel(t);
        size_t bytes = (size_t)n * sizeof(float);
        st->m[p] = (float*)slate_arena_alloc(state_arena, bytes, 16);
        st->v[p] = (float*)slate_arena_alloc(state_arena, bytes, 16);
    }

    opt->step = adamw_step;
    opt->zero_grad = adamw_zero_grad;
    opt->set_lr = adamw_set_lr;
    opt->destroy = adamw_destroy;
    opt->user_data = st;
    return opt;
}
