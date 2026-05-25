// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// matmul_bf16.c — bf16 inputs, fp32 accumulator, fp32 output matmul.
// This is the standard "mixed precision" path used in transformer training:
// activations and weights live in bf16 (half the memory, full dynamic range
// of fp32), but matmul accumulates in fp32 to avoid catastrophic cancellation.

#include "slate/ops.h"
#include "slate/tensor.h"
#include "slate/autograd.h"
#include "slate/runtime.h"
#include "slate/precision.h"
#include <stdint.h>
#include <string.h>

slate_tensor_t* slate_op_matmul_bf16(slate_graph_ctx_t* ctx,
                                      slate_tensor_t* a,
                                      slate_tensor_t* b);

static void mmul_backward(slate_graph_node_t* node) {
    slate_tensor_t* a = node->inputs[0];
    slate_tensor_t* b = node->inputs[1];
    slate_tensor_t* out = node->output;
    int64_t M = a->shape[0], K = a->shape[1], N = b->shape[1];
    const float* dy = (const float*)out->grad;

    if (a->requires_grad && a->grad) {
        // a is bf16; its grad buffer is fp32 (master grad).
        float* da = (float*)a->grad;
        const uint16_t* pb = (const uint16_t*)b->data;
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t k = 0; k < K; ++k) {
                float acc = 0;
                for (int64_t j = 0; j < N; ++j) {
                    float bb = slate_bf16_to_f32(pb[k * N + j]);
                    acc += dy[i * N + j] * bb;
                }
                da[i * K + k] += acc;
            }
        }
    }
    if (b->requires_grad && b->grad) {
        float* db = (float*)b->grad;
        const uint16_t* pa = (const uint16_t*)a->data;
        for (int64_t k = 0; k < K; ++k) {
            for (int64_t j = 0; j < N; ++j) {
                float acc = 0;
                for (int64_t i = 0; i < M; ++i) {
                    float aa = slate_bf16_to_f32(pa[i * K + k]);
                    acc += aa * dy[i * N + j];
                }
                db[k * N + j] += acc;
            }
        }
    }
}

slate_tensor_t* slate_op_matmul_bf16(slate_graph_ctx_t* ctx,
                                      slate_tensor_t* a, slate_tensor_t* b) {
    if (!ctx || !a || !b) return NULL;
    if (a->dtype != SLATE_DTYPE_BF16 || b->dtype != SLATE_DTYPE_BF16) return NULL;
    if (a->n_dims != 2 || b->n_dims != 2) return NULL;
    if (a->shape[1] != b->shape[0]) return NULL;
    int64_t M = a->shape[0], K = a->shape[1], N = b->shape[1];
    int64_t os[2] = {M, N};
    // Output in fp32 (this matches modern transformer training: forward in bf16,
    // loss/output stay fp32 for accumulator stability).
    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32, 2, os, false);
    const uint16_t* pa = (const uint16_t*)a->data;
    const uint16_t* pb = (const uint16_t*)b->data;
    float* po = (float*)out->data;
    memset(po, 0, (size_t)(M * N) * sizeof(float));
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            float aa = slate_bf16_to_f32(pa[i * K + k]);
            for (int64_t j = 0; j < N; ++j) {
                float bb = slate_bf16_to_f32(pb[k * N + j]);
                po[i * N + j] += aa * bb;  // fp32 accumulator
            }
        }
    }
    slate_tensor_t* inputs[2] = {a, b};
    slate_graph_node_t* node = slate_graph_record(ctx, "matmul_bf16", inputs, 2, out, mmul_backward);
    if (node && out->requires_grad && !out->grad) {
        out->grad = slate_arena_alloc(ctx->scratch_arena,
                                       (size_t)slate_tensor_numel(out) * sizeof(float), 16);
    }
    return out;
}
