// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#include "slate/transformer.h"
#include "slate/tensor.h"
#include "slate/ops.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct emb_mod {
    slate_module_t base;
    slate_tensor_t* W;
    int vocab, d_model;
} emb_mod_t;

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

static slate_tensor_t* emb_fwd(slate_module_t* self, slate_graph_ctx_t* ctx, slate_tensor_t* x) {
    emb_mod_t* m = (emb_mod_t*)self;
    return slate_op_embedding(ctx, m->W, x);
}
static void emb_reg(slate_module_t* self, slate_param_set_t* ps) {
    emb_mod_t* m = (emb_mod_t*)self;
    slate_param_set_add(ps, m->W);
}
static void emb_destroy(slate_module_t* self) { free(self); }

slate_module_t* slate_module_embedding_new(slate_arena_t* params,
                                            int vocab, int d_model,
                                            uint64_t seed) {
    emb_mod_t* m = (emb_mod_t*)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->base.name = "Embedding";
    m->base.forward = emb_fwd;
    m->base.register_params = emb_reg;
    m->base.destroy = emb_destroy;
    m->vocab = vocab; m->d_model = d_model;
    int64_t s[2] = {vocab, d_model};
    m->W = slate_tensor_new(params, SLATE_DTYPE_F32, 2, s, true);
    float stddev = 0.02f;
    xrng_t rng; xseed(&rng, seed);
    int64_t n = (int64_t)vocab * d_model;
    float* p = (float*)m->W->data;
    for (int64_t i = 0; i < n; ++i) {
        // Box-Muller for normal init.
        float u1 = xunif(&rng); if (u1 < 1e-7f) u1 = 1e-7f;
        float u2 = xunif(&rng);
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979f * u2);
        p[i] = z * stddev;
    }
    return &m->base;
}
