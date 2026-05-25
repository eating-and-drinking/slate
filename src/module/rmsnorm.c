// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#include "slate/transformer.h"
#include "slate/tensor.h"
#include "slate/ops.h"
#include <stdlib.h>

typedef struct rms_mod {
    slate_module_t base;
    slate_tensor_t* w;
    float eps;
} rms_mod_t;

static slate_tensor_t* rms_fwd(slate_module_t* self, slate_graph_ctx_t* ctx, slate_tensor_t* x) {
    rms_mod_t* m = (rms_mod_t*)self;
    return slate_op_rms_norm(ctx, x, m->w, m->eps);
}
static void rms_reg(slate_module_t* self, slate_param_set_t* ps) {
    rms_mod_t* m = (rms_mod_t*)self;
    slate_param_set_add(ps, m->w);
}
static void rms_destroy(slate_module_t* self) { free(self); }

slate_module_t* slate_module_rmsnorm_new(slate_arena_t* params, int d, float eps) {
    rms_mod_t* m = (rms_mod_t*)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->base.name = "RMSNorm";
    m->base.forward = rms_fwd;
    m->base.register_params = rms_reg;
    m->base.destroy = rms_destroy;
    m->eps = eps;
    int64_t s[1] = {d};
    m->w = slate_tensor_new(params, SLATE_DTYPE_F32, 1, s, true);
    float* p = (float*)m->w->data;
    for (int i = 0; i < d; ++i) p[i] = 1.0f;  // init to 1
    return &m->base;
}
