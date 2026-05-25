// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// attention.c — single-head causal self-attention.
// Multi-head + RoPE are M3+. For M2 a single head is sufficient to demonstrate
// the Transformer architecture works.

#include "slate/transformer.h"
#include "slate/tensor.h"
#include "slate/ops.h"
#include <math.h>
#include <stdlib.h>

typedef struct attn_mod {
    slate_module_t base;
    slate_tensor_t* Wq;
    slate_tensor_t* Wk;
    slate_tensor_t* Wv;
    slate_tensor_t* Wo;
    int d;
} attn_mod_t;

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

static slate_tensor_t* attn_fwd(slate_module_t* self, slate_graph_ctx_t* ctx, slate_tensor_t* x) {
    attn_mod_t* m = (attn_mod_t*)self;
    int d = m->d;
    // Q, K, V: [B, T, D]
    slate_tensor_t* q = slate_op_linear3d(ctx, x, m->Wq);
    slate_tensor_t* k = slate_op_linear3d(ctx, x, m->Wk);
    slate_tensor_t* v = slate_op_linear3d(ctx, x, m->Wv);
    // scores = q @ k^T : [B, T, T]
    slate_tensor_t* k_t = slate_op_transpose_last2(ctx, k);
    slate_tensor_t* scores = slate_op_bmm(ctx, q, k_t);
    // mask + scale
    float scale = 1.0f / sqrtf((float)d);
    slate_tensor_t* masked = slate_op_causal_mask(ctx, scores, scale);
    // attn_weights = softmax(masked) : [B, T, T]
    slate_tensor_t* attn_weights = slate_op_softmax(ctx, masked);
    // out = attn_weights @ v : [B, T, D]
    slate_tensor_t* out = slate_op_bmm(ctx, attn_weights, v);
    // output projection
    slate_tensor_t* y = slate_op_linear3d(ctx, out, m->Wo);
    return y;
}

static void attn_reg(slate_module_t* self, slate_param_set_t* ps) {
    attn_mod_t* m = (attn_mod_t*)self;
    slate_param_set_add(ps, m->Wq);
    slate_param_set_add(ps, m->Wk);
    slate_param_set_add(ps, m->Wv);
    slate_param_set_add(ps, m->Wo);
}
static void attn_destroy(slate_module_t* self) { free(self); }

slate_module_t* slate_module_attention_new(slate_arena_t* params, int d, uint64_t seed) {
    attn_mod_t* m = (attn_mod_t*)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->base.name = "Attention";
    m->base.forward = attn_fwd;
    m->base.register_params = attn_reg;
    m->base.destroy = attn_destroy;
    m->d = d;
    int64_t s[2] = {d, d};
    m->Wq = slate_tensor_new(params, SLATE_DTYPE_F32, 2, s, true); init_kaiming(m->Wq, d, seed + 1);
    m->Wk = slate_tensor_new(params, SLATE_DTYPE_F32, 2, s, true); init_kaiming(m->Wk, d, seed + 2);
    m->Wv = slate_tensor_new(params, SLATE_DTYPE_F32, 2, s, true); init_kaiming(m->Wv, d, seed + 3);
    m->Wo = slate_tensor_new(params, SLATE_DTYPE_F32, 2, s, true); init_kaiming(m->Wo, d, seed + 4);
    return &m->base;
}
