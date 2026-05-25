// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// sgd.c — vanilla SGD with optional momentum.

#include "slate/optim.h"
#include "slate/tensor.h"
#include "slate/module.h"

#include <stdlib.h>
#include <string.h>

typedef struct sgd_state {
    slate_param_set_t* params;
    float lr;
    float momentum;
    float** velocity;   // per-parameter, NULL if momentum == 0
} sgd_state_t;

static void sgd_step(slate_optimizer_t* self) {
    sgd_state_t* st = (sgd_state_t*)self->user_data;
    for (int p = 0; p < st->params->n_params; ++p) {
        slate_tensor_t* t = st->params->params[p];
        if (!t || !t->grad) continue;
        float* w = (float*)t->data;
        const float* g = (const float*)t->grad;
        int64_t n = slate_tensor_numel(t);

        if (st->momentum > 0.0f && st->velocity && st->velocity[p]) {
            float* v = st->velocity[p];
            for (int64_t i = 0; i < n; ++i) {
                v[i] = st->momentum * v[i] + g[i];
                w[i] -= st->lr * v[i];
            }
        } else {
            for (int64_t i = 0; i < n; ++i) {
                w[i] -= st->lr * g[i];
            }
        }
    }
}

static void sgd_zero_grad(slate_optimizer_t* self) {
    sgd_state_t* st = (sgd_state_t*)self->user_data;
    for (int p = 0; p < st->params->n_params; ++p) {
        slate_tensor_zero_grad(st->params->params[p]);
    }
}

static void sgd_set_lr(slate_optimizer_t* self, float lr) {
    sgd_state_t* st = (sgd_state_t*)self->user_data;
    st->lr = lr;
}

static void sgd_destroy(slate_optimizer_t* self) {
    if (!self) return;
    sgd_state_t* st = (sgd_state_t*)self->user_data;
    if (st && st->velocity) free(st->velocity);
    free(st);
    free(self);
}

slate_optimizer_t* slate_optimizer_sgd_new(slate_arena_t* state_arena,
                                            slate_param_set_t* params,
                                            float learning_rate,
                                            float momentum) {
    slate_optimizer_t* opt = (slate_optimizer_t*)calloc(1, sizeof(*opt));
    sgd_state_t* st = (sgd_state_t*)calloc(1, sizeof(*st));
    if (!opt || !st) { free(opt); free(st); return NULL; }

    st->params = params;
    st->lr = learning_rate;
    st->momentum = momentum;
    if (momentum > 0.0f) {
        st->velocity = (float**)calloc((size_t)params->n_params, sizeof(float*));
        for (int p = 0; p < params->n_params; ++p) {
            slate_tensor_t* t = params->params[p];
            int64_t n = slate_tensor_numel(t);
            st->velocity[p] = (float*)slate_arena_alloc(state_arena,
                                                        (size_t)n * sizeof(float),
                                                        16);
        }
    }

    opt->step = sgd_step;
    opt->zero_grad = sgd_zero_grad;
    opt->set_lr = sgd_set_lr;
    opt->destroy = sgd_destroy;
    opt->user_data = st;
    return opt;
}
