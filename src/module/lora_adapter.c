// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// lora_adapter.c — LoRA wrapping a frozen base linear projection.
//
//   y = (x @ W_base)  +  (alpha / r) * (x @ A) @ B
//
// W_base [in, out] is held with requires_grad=false (we still allocate its
// data buffer here but the trainer will skip it during optimizer.step()).
// A [in, r] is initialized to ~N(0, init_std). B [r, out] is initialized to
// zero. Result: at step 0 the LoRA delta is zero and the model behaves
// identically to the base.

#include "slate/transformer.h"
#include "slate/tensor.h"
#include "slate/ops.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

slate_module_t* slate_module_lora_new(slate_arena_t* params,
                                       int in_features, int out_features,
                                       int rank, float alpha,
                                       const float* base_weight,
                                       uint64_t seed);

typedef struct lora_mod {
    slate_module_t base;
    slate_tensor_t* W_base;  // [in, out], requires_grad=false
    slate_tensor_t* A;       // [in, rank], trainable
    slate_tensor_t* B;       // [rank, out], trainable
    float scale;             // alpha / rank
    int rank;
} lora_mod_t;

// xoshiro256** RNG, again (we keep this self-contained per module).
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

static slate_tensor_t* lora_fwd(slate_module_t* self, slate_graph_ctx_t* ctx, slate_tensor_t* x) {
    lora_mod_t* m = (lora_mod_t*)self;
    // x is either [B, in] (2D) or [B, T, in] (3D). Dispatch.
    slate_tensor_t* y_base;
    slate_tensor_t* xa;
    if (x->n_dims == 2) {
        y_base = slate_op_matmul(ctx, x, m->W_base);
        xa = slate_op_matmul(ctx, x, m->A);
    } else {
        y_base = slate_op_linear3d(ctx, x, m->W_base);
        xa = slate_op_linear3d(ctx, x, m->A);
    }
    slate_tensor_t* delta;
    if (x->n_dims == 2) {
        slate_tensor_t* xab = slate_op_matmul(ctx, xa, m->B);
        delta = slate_op_scale(ctx, xab, m->scale);
    } else {
        slate_tensor_t* xab = slate_op_linear3d(ctx, xa, m->B);
        delta = slate_op_scale(ctx, xab, m->scale);
    }
    return slate_op_add(ctx, y_base, delta);
}

static void lora_reg(slate_module_t* self, slate_param_set_t* ps) {
    lora_mod_t* m = (lora_mod_t*)self;
    // CRUCIAL: only A and B go to the optimizer. W_base is frozen.
    slate_param_set_add(ps, m->A);
    slate_param_set_add(ps, m->B);
}
static void lora_destroy(slate_module_t* self) { free(self); }

slate_module_t* slate_module_lora_new(slate_arena_t* params,
                                       int in_features, int out_features,
                                       int rank, float alpha,
                                       const float* base_weight,
                                       uint64_t seed) {
    if (rank <= 0 || rank > in_features || rank > out_features) return NULL;
    lora_mod_t* m = (lora_mod_t*)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->base.name = "LoRA";
    m->base.forward = lora_fwd;
    m->base.register_params = lora_reg;
    m->base.destroy = lora_destroy;
    m->rank = rank;
    m->scale = alpha / (float)rank;

    int64_t Ws[2] = {in_features, out_features};
    int64_t As[2] = {in_features, rank};
    int64_t Bs[2] = {rank, out_features};
    // requires_grad=false for base — the trainer will skip this in optimizer.step.
    m->W_base = slate_tensor_new(params, SLATE_DTYPE_F32, 2, Ws, false);
    m->A      = slate_tensor_new(params, SLATE_DTYPE_F32, 2, As, true);
    m->B      = slate_tensor_new(params, SLATE_DTYPE_F32, 2, Bs, true);
    if (!m->W_base || !m->A || !m->B) { free(m); return NULL; }

    // Copy base weights.
    if (base_weight) {
        memcpy(m->W_base->data, base_weight,
               (size_t)in_features * out_features * sizeof(float));
    }

    // A init: small normal. B init: zero (so initial delta = 0).
    xrng_t rng; xseed(&rng, seed);
    int64_t na = (int64_t)in_features * rank;
    float std = 1.0f / sqrtf((float)in_features);
    float* pa = (float*)m->A->data;
    for (int64_t i = 0; i < na; ++i) {
        float u1 = xunif(&rng); if (u1 < 1e-7f) u1 = 1e-7f;
        float u2 = xunif(&rng);
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
        pa[i] = z * std;
    }
    // B already zero from arena calloc-equivalent (slate_arena_alloc zeros).
    return &m->base;
}
