// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// rope.c — Rotary Position Embedding (RoPE) for LLaMA-class models.
//
// For each (batch, time, head) the head_dim values are split into two
// halves of size hd/2.  Each pair (x[i], x[i + hd/2]) is rotated by
// angle theta_p,i = position[t] * theta_base^(-2i / hd) for i in
// [0, hd/2).
//
// This is the GGML "non-interleaved" convention (also called "neox"
// in some codebases) — the first half of head_dim gets the cosine
// terms, the second half gets the sine terms.  LLaMA-2, LLaMA-3,
// Mistral, Qwen2 all use this layout.
//
// Forward:
//   y[..., i]        =  x[..., i] * cos_p_i - x[..., i + hd/2] * sin_p_i
//   y[..., i + hd/2] =  x[..., i] * sin_p_i + x[..., i + hd/2] * cos_p_i
//
// Backward: rotation by -theta_p,i (i.e. swap sin sign).

#include "slate/ops.h"
#include "slate/tensor.h"
#include "slate/autograd.h"
#include "slate/runtime.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const int32_t* positions;  // [T] (alias into input tensor)
    float theta_base;
    int64_t B, T, n_heads, head_dim;
} rope_state_t;

static void rope_forward_apply(float* y, const float* x,
                                 const int32_t* positions,
                                 int64_t B, int64_t T, int64_t H, int64_t hd,
                                 float theta_base) {
    int half = (int)(hd / 2);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            float pos = (float)positions[t];
            for (int64_t h = 0; h < H; ++h) {
                const float* x_row = x + ((b * T + t) * H + h) * hd;
                float*       y_row = y + ((b * T + t) * H + h) * hd;
                for (int i = 0; i < half; ++i) {
                    float inv_freq = powf(theta_base, -2.0f * (float)i / (float)hd);
                    float angle = pos * inv_freq;
                    float c = cosf(angle);
                    float s = sinf(angle);
                    float a = x_row[i];
                    float b2 = x_row[i + half];
                    y_row[i]        = a * c - b2 * s;
                    y_row[i + half] = a * s + b2 * c;
                }
            }
        }
    }
}

static void rope_backward(slate_graph_node_t* node) {
    slate_tensor_t* x = node->inputs[0];
    slate_tensor_t* out = node->output;
    if (!x->requires_grad || !x->grad) return;
    rope_state_t* st = (rope_state_t*)node->user_data;
    const float* d_y = (const float*)out->grad;
    float* d_x = (float*)x->grad;
    int64_t B = st->B, T = st->T, H = st->n_heads, hd = st->head_dim;
    int half = (int)(hd / 2);
    // Backward: rotate dy by -angle and add to dx
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            float pos = (float)st->positions[t];
            for (int64_t h = 0; h < H; ++h) {
                const float* dy_row = d_y + ((b * T + t) * H + h) * hd;
                float*       dx_row = d_x + ((b * T + t) * H + h) * hd;
                for (int i = 0; i < half; ++i) {
                    float inv_freq = powf(st->theta_base, -2.0f * (float)i / (float)hd);
                    float angle = pos * inv_freq;
                    float c = cosf(angle);
                    float s = sinf(angle);
                    // y_i = x_i*c - x_j*s,  y_j = x_i*s + x_j*c
                    // dx_i = dy_i*c + dy_j*s
                    // dx_j = -dy_i*s + dy_j*c
                    float dyi = dy_row[i];
                    float dyj = dy_row[i + half];
                    dx_row[i]        +=  dyi * c + dyj * s;
                    dx_row[i + half] += -dyi * s + dyj * c;
                }
            }
        }
    }
}

slate_tensor_t* slate_op_rope(slate_graph_ctx_t* ctx,
                               slate_tensor_t* x,
                               slate_tensor_t* positions,
                               float theta_base) {
    if (!ctx || !x || !positions) return NULL;
    if (x->n_dims != 4 || x->dtype != SLATE_DTYPE_F32) return NULL;
    if (positions->n_dims != 1 || positions->dtype != SLATE_DTYPE_I32) return NULL;
    int64_t B  = x->shape[0];
    int64_t T  = x->shape[1];
    int64_t H  = x->shape[2];
    int64_t hd = x->shape[3];
    if (hd % 2 != 0) return NULL;
    if (positions->shape[0] != T) return NULL;

    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32,
                                            4, x->shape, false);
    if (!out) return NULL;

    rope_forward_apply((float*)out->data,
                        (const float*)x->data,
                        (const int32_t*)positions->data,
                        B, T, H, hd, theta_base);

    slate_tensor_t* inputs[1] = { x };
    slate_graph_node_t* node = slate_graph_record(ctx, "rope", inputs, 1, out,
                                                   rope_backward);
    if (node) {
        rope_state_t* st = (rope_state_t*)slate_arena_alloc(ctx->scratch_arena,
                                                              sizeof(*st), 16);
        st->positions = (const int32_t*)positions->data;
        st->theta_base = theta_base;
        st->B = B; st->T = T; st->n_heads = H; st->head_dim = hd;
        node->user_data = st;
        if (out->requires_grad && !out->grad) {
            out->grad = slate_arena_alloc(ctx->scratch_arena,
                                           (size_t)slate_tensor_numel(out) * sizeof(float),
                                           16);
        }
    }
    return out;
}
