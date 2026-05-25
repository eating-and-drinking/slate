// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// linear3d.c — out[b, t, :] = x[b, t, :] @ W. x [B, T, D_in], W [D_in, D_out].

#include "slate/ops.h"
#include "slate/tensor.h"
#include "slate/autograd.h"
#include "slate/runtime.h"
#include <string.h>

slate_tensor_t* slate_op_linear3d(slate_graph_ctx_t*, slate_tensor_t*, slate_tensor_t*);

static void linear3d_backward(slate_graph_node_t* node) {
    slate_tensor_t* x = node->inputs[0];
    slate_tensor_t* W = node->inputs[1];
    slate_tensor_t* out = node->output;
    int64_t B = x->shape[0], T = x->shape[1], Din = x->shape[2];
    int64_t Dout = W->shape[1];
    const float* dy = (const float*)out->grad;
    const float* px = (const float*)x->data;
    const float* pw = (const float*)W->data;
    int64_t BT = B * T;

    if (x->requires_grad && x->grad) {
        float* dx = (float*)x->grad;
        // dx[bt, k] += sum_o dy[bt, o] * W[k, o]
        for (int64_t bt = 0; bt < BT; ++bt) {
            const float* dy_r = dy + bt * Dout;
            float* dx_r = dx + bt * Din;
            for (int64_t k = 0; k < Din; ++k) {
                float acc = 0.0f;
                const float* w_row = pw + k * Dout;
                for (int64_t o = 0; o < Dout; ++o) acc += dy_r[o] * w_row[o];
                dx_r[k] += acc;
            }
        }
    }
    if (W->requires_grad && W->grad) {
        float* dW = (float*)W->grad;
        // dW[k, o] += sum_bt x[bt, k] * dy[bt, o]
        for (int64_t k = 0; k < Din; ++k) {
            for (int64_t o = 0; o < Dout; ++o) {
                float acc = 0.0f;
                for (int64_t bt = 0; bt < BT; ++bt) {
                    acc += px[bt * Din + k] * dy[bt * Dout + o];
                }
                dW[k * Dout + o] += acc;
            }
        }
    }
}

slate_tensor_t* slate_op_linear3d(slate_graph_ctx_t* ctx,
                                   slate_tensor_t* x,
                                   slate_tensor_t* W) {
    if (!ctx || !x || !W) return NULL;
    if (x->n_dims != 3 || W->n_dims != 2) return NULL;
    if (x->shape[2] != W->shape[0]) return NULL;
    int64_t B = x->shape[0], T = x->shape[1], Din = x->shape[2];
    int64_t Dout = W->shape[1];
    int64_t os[3] = {B, T, Dout};
    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32, 3, os, false);
    if (!out) return NULL;
    const float* px = (const float*)x->data;
    const float* pw = (const float*)W->data;
    float* po = (float*)out->data;
    int64_t BT = B * T;
    memset(po, 0, (size_t)(BT * Dout) * sizeof(float));
    for (int64_t bt = 0; bt < BT; ++bt) {
        const float* x_r = px + bt * Din;
        float* o_r = po + bt * Dout;
        for (int64_t k = 0; k < Din; ++k) {
            float xk = x_r[k];
            const float* w_row = pw + k * Dout;
            for (int64_t o = 0; o < Dout; ++o) o_r[o] += xk * w_row[o];
        }
    }
    slate_tensor_t* inputs[2] = {x, W};
    slate_graph_node_t* node = slate_graph_record(ctx, "linear3d", inputs, 2, out, linear3d_backward);
    if (node && out->requires_grad && !out->grad) {
        out->grad = slate_arena_alloc(ctx->scratch_arena, (size_t)slate_tensor_numel(out) * sizeof(float), 16);
    }
    return out;
}
