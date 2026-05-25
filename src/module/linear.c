// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// linear.c — y = x @ W + b.

#include "slate/module.h"
#include "slate/tensor.h"
#include "slate/ops.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct linear_module {
    slate_module_t base;
    slate_tensor_t* W;
    slate_tensor_t* b;  // may be NULL
    int in_features;
    int out_features;
} linear_module_t;

// xoshiro256** for portable, repeatable initialization.
static uint64_t xrot(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
typedef struct { uint64_t s[4]; } xoshiro_t;
static void xoshiro_seed(xoshiro_t* r, uint64_t seed) {
    // SplitMix64 to scatter the seed.
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 4; ++i) {
        z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ULL;
        z ^= z >> 27; z *= 0x94D049BB133111EBULL;
        z ^= z >> 31;
        r->s[i] = z;
    }
}
static uint64_t xoshiro_next(xoshiro_t* r) {
    uint64_t result = xrot(r->s[1] * 5, 7) * 9;
    uint64_t t = r->s[1] << 17;
    r->s[2] ^= r->s[0];
    r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2];
    r->s[0] ^= r->s[3];
    r->s[2] ^= t;
    r->s[3] = xrot(r->s[3], 45);
    return result;
}
static float xoshiro_uniform(xoshiro_t* r) {
    uint64_t u = xoshiro_next(r) >> 11;  // 53 random bits
    return (float)((double)u / (double)(1ULL << 53));
}

static slate_tensor_t* linear_forward(slate_module_t* self,
                                       slate_graph_ctx_t* ctx,
                                       slate_tensor_t* x) {
    linear_module_t* m = (linear_module_t*)self;
    slate_tensor_t* y = slate_op_matmul(ctx, x, m->W);
    if (m->b) {
        // M1: proper bias via add_bias op (broadcast + correct grad reduction).
        y = slate_op_add_bias(ctx, y, m->b);
    }
    return y;
}

static void linear_register_params(slate_module_t* self, slate_param_set_t* ps) {
    linear_module_t* m = (linear_module_t*)self;
    slate_param_set_add(ps, m->W);
    if (m->b) slate_param_set_add(ps, m->b);
}

static void linear_destroy(slate_module_t* self) {
    free(self);  // Tensors live on param_arena; nothing else to free here.
}

slate_module_t* slate_module_linear_new(slate_arena_t* param_arena,
                                        int in_features,
                                        int out_features,
                                        bool with_bias,
                                        uint64_t init_seed) {
    linear_module_t* m = (linear_module_t*)calloc(1, sizeof(*m));
    if (!m) return NULL;

    m->base.name = "Linear";
    m->base.forward = linear_forward;
    m->base.register_params = linear_register_params;
    m->base.destroy = linear_destroy;
    m->in_features = in_features;
    m->out_features = out_features;

    int64_t Wshape[2] = {in_features, out_features};
    m->W = slate_tensor_new(param_arena, SLATE_DTYPE_F32, 2, Wshape, true);
    if (!m->W) { free(m); return NULL; }

    // Kaiming-uniform init for ReLU-style nets:
    //   bound = sqrt(6 / fan_in)
    float bound = sqrtf(6.0f / (float)in_features);
    xoshiro_t rng; xoshiro_seed(&rng, init_seed);
    int64_t n = slate_tensor_numel(m->W);
    float* w = (float*)m->W->data;
    for (int64_t i = 0; i < n; ++i) {
        float u = xoshiro_uniform(&rng);   // [0, 1)
        w[i] = (u * 2.0f - 1.0f) * bound;  // [-bound, bound]
    }

    if (with_bias) {
        int64_t bshape[1] = {out_features};
        m->b = slate_tensor_new(param_arena, SLATE_DTYPE_F32, 1, bshape, true);
        // Zero-init bias.
    }

    return &m->base;
}
