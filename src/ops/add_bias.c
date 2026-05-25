// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// add_bias.c — y[..., c] = x[..., c] + b[c].  AVX2 SIMD broadcast-add.

#include "slate/ops.h"
#include "slate/tensor.h"
#include "slate/autograd.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

static void add_bias_backward(slate_graph_node_t* node) {
    slate_tensor_t* x = node->inputs[0];
    slate_tensor_t* b = node->inputs[1];
    slate_tensor_t* out = node->output;
    int64_t C = b->shape[0];
    int64_t N = slate_tensor_numel(out) / C;
    const float* d_out = (const float*)out->grad;

    if (x->requires_grad && x->grad) {
        float* d_x = (float*)x->grad;
        int64_t n = slate_tensor_numel(out);
#if defined(__AVX2__)
        int64_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 dxv = _mm256_loadu_ps(d_x + i);
            __m256 dyv = _mm256_loadu_ps(d_out + i);
            _mm256_storeu_ps(d_x + i, _mm256_add_ps(dxv, dyv));
        }
        for (; i < n; ++i) d_x[i] += d_out[i];
#else
        for (int64_t i = 0; i < n; ++i) d_x[i] += d_out[i];
#endif
    }
    if (b->requires_grad && b->grad) {
        float* d_b = (float*)b->grad;
        for (int64_t i = 0; i < N; ++i) {
            const float* row = d_out + i * C;
#if defined(__AVX2__)
            int64_t c = 0;
            for (; c + 8 <= C; c += 8) {
                __m256 dbv = _mm256_loadu_ps(d_b + c);
                __m256 rv  = _mm256_loadu_ps(row + c);
                _mm256_storeu_ps(d_b + c, _mm256_add_ps(dbv, rv));
            }
            for (; c < C; ++c) d_b[c] += row[c];
#else
            for (int64_t c = 0; c < C; ++c) d_b[c] += row[c];
#endif
        }
    }
}

slate_tensor_t* slate_op_add_bias(slate_graph_ctx_t* ctx,
                                  slate_tensor_t* x,
                                  slate_tensor_t* b) {
    if (!ctx || !x || !b) return NULL;
    if (x->dtype != SLATE_DTYPE_F32 || b->dtype != SLATE_DTYPE_F32) return NULL;
    if (b->n_dims != 1) return NULL;
    if (x->shape[x->n_dims - 1] != b->shape[0]) return NULL;

    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32,
                                            x->n_dims, x->shape, false);
    if (!out) return NULL;

    int64_t C = b->shape[0];
    int64_t N = slate_tensor_numel(out) / C;
    const float* px = (const float*)x->data;
    const float* pb = (const float*)b->data;
    float* po = (float*)out->data;
    for (int64_t n = 0; n < N; ++n) {
        const float* row = px + n * C;
        float* orow = po + n * C;
#if defined(__AVX2__)
        int64_t c = 0;
        for (; c + 8 <= C; c += 8) {
            __m256 xv = _mm256_loadu_ps(row + c);
            __m256 bv = _mm256_loadu_ps(pb  + c);
            _mm256_storeu_ps(orow + c, _mm256_add_ps(xv, bv));
        }
        for (; c < C; ++c) orow[c] = row[c] + pb[c];
#else
        for (int64_t c = 0; c < C; ++c) orow[c] = row[c] + pb[c];
#endif
    }

    slate_tensor_t* inputs[2] = {x, b};
    slate_graph_node_t* node = slate_graph_record(ctx, "add_bias", inputs, 2, out,
                                                   add_bias_backward);
    if (node && out->requires_grad && !out->grad) {
        out->grad = slate_arena_alloc(ctx->scratch_arena,
                                       (size_t)slate_tensor_numel(out) * sizeof(float),
                                       16);
    }
    return out;
}
