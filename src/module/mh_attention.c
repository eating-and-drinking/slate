// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// mh_attention.c — multi-head causal self-attention.
//   x [B, T, D] --Wq/Wk/Wv--> q,k,v [B, T, D]
//   reshape to [B, T, H, Dh], permute to [B, H, T, Dh]
//   scores = q @ k^T : [B, H, T, T] with 4D bmm
//   mask + softmax, then @ v : [B, H, T, Dh]
//   permute back, reshape back, output projection

#include "slate/transformer.h"
#include "slate/tensor.h"
#include "slate/ops.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

slate_module_t* slate_module_mh_attention_new(slate_arena_t* params,
                                               int d_model, int n_heads,
                                               uint64_t seed);

typedef struct mha {
    slate_module_t base;
    slate_tensor_t *Wq, *Wk, *Wv, *Wo;
    int d_model, n_heads, d_head;
} mha_t;

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
static void init_k(slate_tensor_t* W, int fan_in, uint64_t seed) {
    xrng_t rng; xseed(&rng, seed);
    float bound = sqrtf(6.0f / (float)fan_in);
    int64_t n = slate_tensor_numel(W);
    float* p = (float*)W->data;
    for (int64_t i = 0; i < n; ++i) {
        float u = (float)((double)(xnext(&rng) >> 11) / (double)(1ULL << 53));
        p[i] = (u * 2.0f - 1.0f) * bound;
    }
}

static slate_tensor_t* mha_fwd(slate_module_t* self, slate_graph_ctx_t* ctx, slate_tensor_t* x) {
    mha_t* m = (mha_t*)self;
    int H = m->n_heads, Dh = m->d_head;
    int64_t B = x->shape[0], T = x->shape[1], D = x->shape[2];

    slate_tensor_t* q = slate_op_linear3d(ctx, x, m->Wq);
    slate_tensor_t* k = slate_op_linear3d(ctx, x, m->Wk);
    slate_tensor_t* v = slate_op_linear3d(ctx, x, m->Wv);

    // View [B, T, D] -> [B, T, H, Dh]
    int64_t sp4[4] = {B, T, H, Dh};
    slate_tensor_t* q4 = slate_tensor_view(ctx->scratch_arena, q, 4, sp4, NULL);
    slate_tensor_t* k4 = slate_tensor_view(ctx->scratch_arena, k, 4, sp4, NULL);
    slate_tensor_t* v4 = slate_tensor_view(ctx->scratch_arena, v, 4, sp4, NULL);

    // Permute [B, T, H, Dh] -> [B, H, T, Dh]
    slate_tensor_t* qp = slate_op_permute_12(ctx, q4);
    slate_tensor_t* kp = slate_op_permute_12(ctx, k4);
    slate_tensor_t* vp = slate_op_permute_12(ctx, v4);

    // Scores: bmm(qp, kp^T) : [B, H, T, T]
    slate_tensor_t* kpt = slate_op_transpose_last2(ctx, kp);
    slate_tensor_t* scores = slate_op_bmm(ctx, qp, kpt);
    float scale = 1.0f / sqrtf((float)Dh);
    slate_tensor_t* masked = slate_op_causal_mask(ctx, scores, scale);
    slate_tensor_t* attn = slate_op_softmax(ctx, masked);
    slate_tensor_t* ctx_out = slate_op_bmm(ctx, attn, vp);  // [B, H, T, Dh]

    // Permute back [B, H, T, Dh] -> [B, T, H, Dh]
    slate_tensor_t* ctx_perm = slate_op_permute_12(ctx, ctx_out);
    // View [B, T, H, Dh] -> [B, T, D]
    int64_t sp3[3] = {B, T, D};
    slate_tensor_t* merged = slate_tensor_view(ctx->scratch_arena, ctx_perm, 3, sp3, NULL);

    return slate_op_linear3d(ctx, merged, m->Wo);
}

static void mha_reg(slate_module_t* self, slate_param_set_t* ps) {
    mha_t* m = (mha_t*)self;
    slate_param_set_add(ps, m->Wq);
    slate_param_set_add(ps, m->Wk);
    slate_param_set_add(ps, m->Wv);
    slate_param_set_add(ps, m->Wo);
}
static void mha_destroy(slate_module_t* self) { free(self); }

slate_module_t* slate_module_mh_attention_new(slate_arena_t* params,
                                               int d_model, int n_heads,
                                               uint64_t seed) {
    if (d_model % n_heads != 0) return NULL;
    mha_t* m = (mha_t*)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->base.name = "MultiHeadAttention";
    m->base.forward = mha_fwd;
    m->base.register_params = mha_reg;
    m->base.destroy = mha_destroy;
    m->d_model = d_model; m->n_heads = n_heads; m->d_head = d_model / n_heads;
    int64_t s[2] = {d_model, d_model};
    m->Wq = slate_tensor_new(params, SLATE_DTYPE_F32, 2, s, true); init_k(m->Wq, d_model, seed + 1);
    m->Wk = slate_tensor_new(params, SLATE_DTYPE_F32, 2, s, true); init_k(m->Wk, d_model, seed + 2);
    m->Wv = slate_tensor_new(params, SLATE_DTYPE_F32, 2, s, true); init_k(m->Wv, d_model, seed + 3);
    m->Wo = slate_tensor_new(params, SLATE_DTYPE_F32, 2, s, true); init_k(m->Wo, d_model, seed + 4);
    return &m->base;
}
