// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#include "slate/checkpoint.h"
#include "slate/tensor.h"
#include "slate/autograd.h"
#include "slate/runtime.h"
#include <stdlib.h>
#include <string.h>

typedef struct ckpt_state {
    slate_checkpoint_fn fn;
    void* user_data;
    // We save the input *data* (and shape/dtype) into a malloc'd buffer so
    // the outer scratch arena can be reset between forward and backward.
    void* x_data_copy;
    size_t x_nbytes;
    slate_dtype_t x_dtype;
    int x_n_dims;
    int64_t x_shape[SLATE_MAX_DIMS];
    // Sub-arenas for the recompute. Owned by us.
    slate_arena_t* sub_nodes;
    slate_arena_t* sub_scratch;
    // The outer x tensor we should write the gradient back into.
    slate_tensor_t* outer_x;
} ckpt_state_t;

static void ckpt_backward(slate_graph_node_t* node) {
    slate_tensor_t* out = node->output;
    ckpt_state_t* st = (ckpt_state_t*)node->user_data;
    if (!st->outer_x->requires_grad || !st->outer_x->grad) return;

    // Build a fresh sub-context with recording enabled, populate input from
    // the saved copy, re-run fn to materialize the intermediate graph.
    slate_arena_reset(st->sub_nodes);
    slate_arena_reset(st->sub_scratch);
    slate_graph_ctx_t sub;
    slate_graph_ctx_init(&sub, st->sub_nodes, st->sub_scratch);
    sub.training = true;

    slate_tensor_t* x_replay = slate_tensor_new(st->sub_scratch, st->x_dtype,
                                                 st->x_n_dims, st->x_shape, true);
    memcpy(x_replay->data, st->x_data_copy, st->x_nbytes);
    slate_tensor_zero_grad(x_replay);

    slate_tensor_t* y_replay = st->fn(&sub, x_replay, st->user_data);

    // Seed y_replay->grad with the upstream gradient from the outer node.
    if (!y_replay->grad) {
        y_replay->grad = slate_arena_alloc(st->sub_scratch,
                                             (size_t)slate_tensor_numel(y_replay) * sizeof(float),
                                             16);
    }
    memcpy(y_replay->grad, out->grad,
           (size_t)slate_tensor_numel(y_replay) * sizeof(float));

    // Walk the sub-graph backward.
    // We bypass the "scalar loss" check in slate_graph_backward by manually
    // running the same reverse loop here:
    for (int i = sub.n_nodes - 1; i >= 0; --i) {
        if (sub.nodes[i]->backward) sub.nodes[i]->backward(sub.nodes[i]);
    }

    // Copy x_replay->grad into outer_x->grad.
    int64_t n = slate_tensor_numel(x_replay);
    const float* g_in = (const float*)x_replay->grad;
    float* g_out = (float*)st->outer_x->grad;
    for (int64_t i = 0; i < n; ++i) g_out[i] += g_in[i];
}

slate_tensor_t* slate_op_checkpoint(slate_graph_ctx_t* ctx,
                                     slate_tensor_t* x,
                                     slate_checkpoint_fn fn,
                                     void* user_data) {
    if (!ctx || !x || !fn) return NULL;

    // Forward path: run fn in a sub-context with training=FALSE so no
    // intermediates are kept. (The output's data is real; we copy it into
    // the outer scratch for the outer graph to use.)
    slate_arena_t* tmp_nodes = slate_arena_create(4 * 1024 * 1024);
    slate_arena_t* tmp_scratch = slate_arena_create(64 * 1024 * 1024);
    slate_graph_ctx_t sub;
    slate_graph_ctx_init(&sub, tmp_nodes, tmp_scratch);
    sub.training = false;
    slate_tensor_t* y_sub = fn(&sub, x, user_data);
    if (!y_sub) {
        slate_arena_destroy(tmp_nodes); slate_arena_destroy(tmp_scratch);
        return NULL;
    }
    // Copy output into outer scratch arena.
    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, y_sub->dtype,
                                            y_sub->n_dims, y_sub->shape, false);
    size_t y_nbytes = (size_t)slate_tensor_numel(y_sub) * slate_dtype_size(y_sub->dtype);
    memcpy(out->data, y_sub->data, y_nbytes);
    // tmp_nodes/tmp_scratch were inference only — destroy now.
    slate_arena_destroy(tmp_nodes);
    slate_arena_destroy(tmp_scratch);

    // Save the input data outside the arena so it survives ctx_reset
    // between forward and backward.
    size_t x_nbytes = (size_t)slate_tensor_numel(x) * slate_dtype_size(x->dtype);
    ckpt_state_t* st = (ckpt_state_t*)malloc(sizeof(*st));
    st->fn = fn;
    st->user_data = user_data;
    st->x_data_copy = malloc(x_nbytes);
    memcpy(st->x_data_copy, x->data, x_nbytes);
    st->x_nbytes = x_nbytes;
    st->x_dtype = x->dtype;
    st->x_n_dims = x->n_dims;
    for (int i = 0; i < x->n_dims; ++i) st->x_shape[i] = x->shape[i];
    // Sub-arenas survive across iterations — allocated once, reset each backward.
    st->sub_nodes = slate_arena_create(4 * 1024 * 1024);
    st->sub_scratch = slate_arena_create(64 * 1024 * 1024);
    st->outer_x = x;

    // Record the checkpoint node in the outer graph.
    slate_tensor_t* inputs[1] = {x};
    slate_graph_node_t* node = slate_graph_record(ctx, "checkpoint", inputs, 1,
                                                   out, ckpt_backward);
    if (node) {
        node->user_data = st;
        if (!out->grad) {
            out->grad = slate_arena_alloc(ctx->scratch_arena,
                                           (size_t)slate_tensor_numel(out) * sizeof(float),
                                           16);
        }
    } else {
        // Not training; release immediately. (Caller still gets the output.)
        free(st->x_data_copy);
        slate_arena_destroy(st->sub_nodes);
        slate_arena_destroy(st->sub_scratch);
        free(st);
    }
    return out;
}
