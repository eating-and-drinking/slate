// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// graph.c — static computation graph and reverse-mode autodiff.

#include "slate/autograd.h"
#include "slate/error.h"
#include "slate/tensor.h"

#include <string.h>

slate_status_t slate_graph_ctx_init(slate_graph_ctx_t* ctx,
                                    slate_arena_t* node_arena,
                                    slate_arena_t* scratch_arena) {
    if (!ctx || !node_arena || !scratch_arena) return SLATE_ERR_INVALID_ARGUMENT;
    ctx->node_arena = node_arena;
    ctx->scratch_arena = scratch_arena;
    ctx->nodes = NULL;
    ctx->n_nodes = 0;
    ctx->cap_nodes = 0;
    ctx->training = true;
    return SLATE_OK;
}

void slate_graph_ctx_set_training(slate_graph_ctx_t* ctx, bool training) {
    if (ctx) ctx->training = training;
}

void slate_graph_ctx_reset(slate_graph_ctx_t* ctx) {
    if (!ctx) return;
    // node_arena and scratch_arena are both reset; the nodes[] backing storage
    // was allocated from node_arena, so it dies with that reset.
    slate_arena_reset(ctx->node_arena);
    slate_arena_reset(ctx->scratch_arena);
    ctx->nodes = NULL;
    ctx->n_nodes = 0;
    ctx->cap_nodes = 0;
}

// Grow the nodes[] array geometrically. Allocations come from node_arena, so
// "growing" actually means allocating a new larger buffer and copying — the
// old buffer is effectively leaked until the next ctx_reset. This is fine for
// a static graph where node counts are predictable.
static slate_status_t ensure_capacity(slate_graph_ctx_t* ctx, int required) {
    if (ctx->cap_nodes >= required) return SLATE_OK;
    int new_cap = ctx->cap_nodes < 16 ? 16 : ctx->cap_nodes * 2;
    while (new_cap < required) new_cap *= 2;

    slate_graph_node_t** new_arr = (slate_graph_node_t**)slate_arena_alloc(
        ctx->node_arena, (size_t)new_cap * sizeof(slate_graph_node_t*), 16);
    if (!new_arr) return SLATE_ERR_OUT_OF_MEMORY;

    if (ctx->nodes) {
        memcpy(new_arr, ctx->nodes, (size_t)ctx->n_nodes * sizeof(slate_graph_node_t*));
    }
    ctx->nodes = new_arr;
    ctx->cap_nodes = new_cap;
    return SLATE_OK;
}

slate_graph_node_t* slate_graph_record(slate_graph_ctx_t* ctx,
                                       const char* op_name,
                                       slate_tensor_t* const* inputs,
                                       int n_inputs,
                                       slate_tensor_t* output,
                                       slate_backward_fn backward) {
    if (!ctx || !output) return NULL;
    if (n_inputs > SLATE_MAX_OP_INPUTS) return NULL;
    if (!ctx->training) return NULL;

    // Determine if backward needs to run on this node: yes iff any input
    // requires grad. Propagate to output.
    bool any_grad = false;
    for (int i = 0; i < n_inputs; ++i) {
        if (inputs[i] && inputs[i]->requires_grad) { any_grad = true; break; }
    }
    if (!any_grad) {
        return NULL;
    }
    output->requires_grad = true;

    if (ensure_capacity(ctx, ctx->n_nodes + 1) != SLATE_OK) return NULL;

    slate_graph_node_t* n = (slate_graph_node_t*)slate_arena_alloc(
        ctx->node_arena, sizeof(*n), 16);
    if (!n) return NULL;

    n->op_name = op_name;
    n->n_inputs = n_inputs;
    for (int i = 0; i < n_inputs; ++i) n->inputs[i] = inputs[i];
    n->output = output;
    n->backward = backward;
    n->user_data = NULL;
    n->topo_visited = 0;

    output->grad_fn = n;
    ctx->nodes[ctx->n_nodes++] = n;
    return n;
}

slate_status_t slate_graph_backward(slate_graph_ctx_t* ctx, slate_tensor_t* loss) {
    if (!ctx || !loss) return SLATE_ERR_INVALID_ARGUMENT;
    if (!loss->requires_grad || !loss->grad)
        return slate_set_error(SLATE_ERR_NO_GRAD,
                               "backward called on tensor without gradient");

    // Seed the upstream gradient with 1.0 (scalar loss).
    if (slate_tensor_numel(loss) != 1)
        return slate_set_error(SLATE_ERR_INVALID_ARGUMENT,
                               "backward requires a scalar loss");
    if (loss->dtype == SLATE_DTYPE_F32) {
        *(float*)loss->grad = 1.0f;
    } else {
        return SLATE_ERR_NOT_IMPLEMENTED;
    }

    // Zero gradients for all non-loss tensors that participate. Parameter
    // tensors had their grad zeroed by the optimizer; intermediate tensors
    // had their grad zeroed at allocation time.

    // Walk recorded nodes in reverse. Order of insertion is already a
    // topological order because each node's output is produced after its
    // inputs.
    for (int i = ctx->n_nodes - 1; i >= 0; --i) {
        slate_graph_node_t* node = ctx->nodes[i];
        if (node->backward) {
            node->backward(node);
        }
    }
    return SLATE_OK;
}
