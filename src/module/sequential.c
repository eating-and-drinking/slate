// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// sequential.c — chain of modules, applied in order.

#include "slate/module.h"

#include <stdlib.h>
#include <string.h>

typedef struct seq_module {
    slate_module_t base;
    slate_module_t** modules;
    int n_modules;
} seq_module_t;

static slate_tensor_t* seq_forward(slate_module_t* self,
                                    slate_graph_ctx_t* ctx,
                                    slate_tensor_t* x) {
    seq_module_t* s = (seq_module_t*)self;
    slate_tensor_t* y = x;
    for (int i = 0; i < s->n_modules; ++i) {
        y = slate_module_forward(s->modules[i], ctx, y);
        if (!y) return NULL;
    }
    return y;
}

static void seq_register_params(slate_module_t* self, slate_param_set_t* ps) {
    seq_module_t* s = (seq_module_t*)self;
    for (int i = 0; i < s->n_modules; ++i) {
        slate_module_register_params(s->modules[i], ps);
    }
}

static void seq_destroy(slate_module_t* self) {
    seq_module_t* s = (seq_module_t*)self;
    for (int i = 0; i < s->n_modules; ++i) {
        slate_module_destroy(s->modules[i]);
    }
    free(s->modules);
    free(s);
}

slate_module_t* slate_module_sequential_new(slate_arena_t* seq_arena,
                                            slate_module_t** modules,
                                            int n_modules) {
    (void)seq_arena;  // sequential's own storage is small, use heap
    if (!modules || n_modules <= 0) return NULL;

    seq_module_t* s = (seq_module_t*)calloc(1, sizeof(*s));
    if (!s) return NULL;

    s->modules = (slate_module_t**)malloc((size_t)n_modules * sizeof(*modules));
    if (!s->modules) { free(s); return NULL; }
    memcpy(s->modules, modules, (size_t)n_modules * sizeof(*modules));
    s->n_modules = n_modules;

    s->base.name = "Sequential";
    s->base.forward = seq_forward;
    s->base.register_params = seq_register_params;
    s->base.destroy = seq_destroy;
    return &s->base;
}
