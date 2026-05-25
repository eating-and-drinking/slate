#define _POSIX_C_SOURCE 200809L
// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// quantized_lora.c — base linear weight lives quantized in a GGUF file;
// LoRA adapter sits on top with trainable A, B matrices.
//
//   y = matmul(x, dequant(W_quant)) + (alpha/r) * (x @ A) @ B
//
// Each forward dequantizes the base on the fly. This is wasteful per step
// but architecturally honest: the base weights are never modified, and the
// GGUF file is touched only via PROT_READ mmap. For real LoRA fine-tuning
// of 7B models, dequant overhead becomes part of the per-step cost and
// is dominated by the matmul itself.

#include "slate/transformer.h"
#include "slate/tensor.h"
#include "slate/ops.h"
#include "slate/gguf.h"
#include "slate/quant.h"
#include "slate/precision.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

slate_module_t* slate_module_quantized_lora_new(slate_arena_t* params,
                                                 slate_gguf_t* gguf,
                                                 const char* tensor_name,
                                                 int rank, float alpha,
                                                 uint64_t seed);

typedef struct ql {
    slate_module_t base;
    slate_gguf_t* gguf;
    char* tensor_name;
    slate_tensor_t* W_view;    // GGUF view (data points into mmap)
    slate_tensor_t* A;         // trainable [in, rank]
    slate_tensor_t* B;         // trainable [rank, out]
    int in_features, out_features;
    float scale;
} ql_t;

static slate_tensor_t* ql_fwd(slate_module_t* self, slate_graph_ctx_t* ctx, slate_tensor_t* x) {
    ql_t* m = (ql_t*)self;
    // Dequantize base [in, out] into scratch f32.
    int64_t numel = (int64_t)m->in_features * m->out_features;
    int64_t ws[2] = {m->in_features, m->out_features};
    slate_tensor_t* W = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32, 2, ws, false);
    if (m->W_view->dtype == SLATE_DTYPE_Q8_0) {
        slate_dequant_q8_0((float*)W->data, m->W_view->data, numel);
    } else if (m->W_view->dtype == SLATE_DTYPE_Q4_0) {
        slate_dequant_q4_0((float*)W->data, m->W_view->data, numel);
    } else if (m->W_view->dtype == SLATE_DTYPE_F16) {
        slate_f16_to_f32_n((float*)W->data, (const uint16_t*)m->W_view->data, (size_t)numel);
    } else {
        memcpy(W->data, m->W_view->data, (size_t)numel * sizeof(float));
    }

    // Base forward: y_base = x @ W
    slate_tensor_t* y_base = slate_op_matmul(ctx, x, W);
    // LoRA delta: (alpha/r) * (x @ A) @ B
    slate_tensor_t* xa = slate_op_matmul(ctx, x, m->A);
    slate_tensor_t* xab = slate_op_matmul(ctx, xa, m->B);
    slate_tensor_t* delta = slate_op_scale(ctx, xab, m->scale);
    return slate_op_add(ctx, y_base, delta);
}

static void ql_reg(slate_module_t* self, slate_param_set_t* ps) {
    ql_t* m = (ql_t*)self;
    // Only LoRA trainables; the GGUF base is by definition frozen.
    slate_param_set_add(ps, m->A);
    slate_param_set_add(ps, m->B);
}
static void ql_destroy(slate_module_t* self) {
    ql_t* m = (ql_t*)self;
    free(m->tensor_name);
    free(m);
}

// Same Box-Muller LoRA A initialization as the regular LoRA module.
static uint64_t xnext(uint64_t* s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return *s;
}
static float xunif(uint64_t* s) {
    return (float)((double)((xnext(s) >> 11) & ((1ULL << 53) - 1)) / (double)(1ULL << 53));
}

slate_module_t* slate_module_quantized_lora_new(slate_arena_t* params,
                                                 slate_gguf_t* gguf,
                                                 const char* tensor_name,
                                                 int rank, float alpha,
                                                 uint64_t seed) {
    ql_t* m = (ql_t*)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->base.name = "QuantizedLoRA";
    m->base.forward = ql_fwd;
    m->base.register_params = ql_reg;
    m->base.destroy = ql_destroy;
    m->gguf = gguf;
    m->tensor_name = strdup(tensor_name);
    m->W_view = slate_gguf_get_tensor(params, gguf, tensor_name);
    if (!m->W_view || m->W_view->n_dims != 2) { free(m); return NULL; }
    m->in_features = (int)m->W_view->shape[0];
    m->out_features = (int)m->W_view->shape[1];
    m->scale = alpha / (float)rank;

    int64_t As[2] = {m->in_features, rank};
    int64_t Bs[2] = {rank, m->out_features};
    m->A = slate_tensor_new(params, SLATE_DTYPE_F32, 2, As, true);
    m->B = slate_tensor_new(params, SLATE_DTYPE_F32, 2, Bs, true);

    uint64_t rng = seed;
    int64_t na = (int64_t)m->in_features * rank;
    float std = 1.0f / sqrtf((float)m->in_features);
    float* pa = (float*)m->A->data;
    for (int64_t i = 0; i < na; ++i) {
        float u1 = xunif(&rng); if (u1 < 1e-7f) u1 = 1e-7f;
        float u2 = xunif(&rng);
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
        pa[i] = z * std;
    }
    return &m->base;
}
