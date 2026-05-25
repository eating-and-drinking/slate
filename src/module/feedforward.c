// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// feedforward.c — SwiGLU FFN: down(silu(gate(x)) * up(x))

#include "slate/transformer.h"
#include "slate/tensor.h"
#include "slate/ops.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct ffn_mod {
    slate_module_t base;
    slate_tensor_t* Wg, *Wu, *Wd;
    int d, h;
} ffn_mod_t;

static uint64_t xrot(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
typedef struct { uint64_t s[4]; } xrng_t;
static void xseed(xrng_t* r, uint64_t s) {
    uint64_t z = s + 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 4; ++i) {
        z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ULL;
        z ^= z >> 27; z *= 0x94D049BB133111EBULL;
        z ^= z >> 31; r->s[i] = z;
    }
}
static uint64_t xnext(xrng_t* r) {
    uint64_t result = xrot(r->s[1] * 5, 7) * 9;
    uint64_t t = r->s[1] << 17;
    r->s[2] ^= r->s[0]; r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2]; r->s[0] ^= r->s[3];
    r->s[2] ^= t; r->s[3] = xrot(r->s[3], 45);
    return result;
}
static float xunif(xrng_t* r) { return (float)((double)(xnext(r) >> 11) / (double)(1ULL << 53)); }
static void init_kaiming(slate_tensor_t* W, int fan_in, uint64_t seed) {
    float bound = sqrtf(6.0f / (float)fan_in);
    xrng_t rng; xseed(&rng, seed);
    int64_t n = slate_tensor_numel(W);
    float* p = (float*)W->data;
    for (int64_t i = 0; i < n; ++i) p[i] = (xunif(&rng) * 2.0f - 1.0f) * bound;
}

static slate_tensor_t* ffn_fwd(slate_module_t* self, slate_graph_ctx_t* ctx, slate_tensor_t* x) {
    ffn_mod_t* m = (ffn_mod_t*)self;
    slate_tensor_t* g = slate_op_linear3d(ctx, x, m->Wg);
    slate_tensor_t* u = slate_op_linear3d(ctx, x, m->Wu);
    slate_tensor_t* gs = slate_op_silu(ctx, g);
    slate_tensor_t* gu = slate_op_mul(ctx, gs, u);
    return slate_op_linear3d(ctx, gu, m->Wd);
}
static void ffn_reg(slate_module_t* self, slate_param_set_t* ps) {
    ffn_mod_t* m = (ffn_mod_t*)self;
    slate_param_set_add(ps, m->Wg);
    slate_param_set_add(ps, m->Wu);
    slate_param_set_add(ps, m->Wd);
}
static void ffn_destroy(slate_module_t* self) { free(self); }

slate_module_t* slate_module_ffn_new(slate_arena_t* params, int d, int h, uint64_t seed) {
    ffn_mod_t* m = (ffn_mod_t*)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->base.name = "FFN";
    m->base.forward = ffn_fwd;
    m->base.register_params = ffn_reg;
    m->base.destroy = ffn_destroy;
    m->d = d; m->h = h;
    int64_t sg[2] = {d, h}, sd[2] = {h, d};
    m->Wg = slate_tensor_new(params, SLATE_DTYPE_F32, 2, sg, true); init_kaiming(m->Wg, d, seed + 1);
    m->Wu = slate_tensor_new(params, SLATE_DTYPE_F32, 2, sg, true); init_kaiming(m->Wu, d, seed + 2);
    m->Wd = slate_tensor_new(params, SLATE_DTYPE_F32, 2, sd, true); init_kaiming(m->Wd, h, seed + 3);
    return &m->base;
}
