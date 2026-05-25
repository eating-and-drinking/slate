// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// autograd.h — L2: static computation graph and reverse-mode backward.
//
// Slate uses a static graph: ops executed against a graph context record
// their inputs and a `backward_fn` callback. `slate_graph_backward` topologically
// walks the recorded ops in reverse and calls each callback, accumulating
// gradients into the participating tensors.

#ifndef SLATE_AUTOGRAD_H
#define SLATE_AUTOGRAD_H

#include "slate/types.h"
#include "slate/runtime.h"
#include "slate/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// A backward callback receives:
//   - `node`: the node it belongs to (use node->user_data for state saved
//             during forward, e.g. cached pre-activation values)
//   - The node's input tensors are accessible via node->inputs[].
//   - The upstream gradient is in node->output->grad.
//   - The callback is responsible for accumulating gradients into the
//     gradient buffers of inputs that have requires_grad == true.
typedef void (*slate_backward_fn)(slate_graph_node_t* node);

struct slate_graph_node {
    const char* op_name;                            // for debugging
    slate_tensor_t* inputs[SLATE_MAX_OP_INPUTS];
    int n_inputs;
    slate_tensor_t* output;
    slate_backward_fn backward;
    void* user_data;                                // saved-for-backward state
    int topo_visited;                               // used by topo sort
};

// Graph context. Owns nothing; everything lives in arenas the caller controls.
//
// Two arenas are needed:
//   - `node_arena`  : holds graph nodes themselves and their saved state. Reset
//                     between forward+backward passes.
//   - `scratch_arena`: where forward ops allocate output buffers. Reset between
//                     iterations.
struct slate_graph_ctx {
    slate_arena_t* node_arena;
    slate_arena_t* scratch_arena;
    slate_graph_node_t** nodes;     // dynamically grown
    int n_nodes;
    int cap_nodes;
    bool training;                  // if false, ops skip graph recording
};

// Initialize a graph context. The two arenas must outlive `ctx`. The
// internal nodes[] buffer is allocated from `node_arena`.
slate_status_t slate_graph_ctx_init(slate_graph_ctx_t* ctx,
                                    slate_arena_t* node_arena,
                                    slate_arena_t* scratch_arena);

// Set or unset training mode. In inference (training=false) the graph is
// not recorded; output tensors do not get a grad_fn.
void slate_graph_ctx_set_training(slate_graph_ctx_t* ctx, bool training);

// Reset the graph for the next iteration: clears the node list and resets
// the node and scratch arenas. Parameter tensors live on a *different*
// arena and are not affected.
void slate_graph_ctx_reset(slate_graph_ctx_t* ctx);

// Internal use by op implementations: append a new node to the graph.
// Initializes inputs and grabs the output tensor's grad_fn pointer.
slate_graph_node_t* slate_graph_record(slate_graph_ctx_t* ctx,
                                       const char* op_name,
                                       slate_tensor_t* const* inputs,
                                       int n_inputs,
                                       slate_tensor_t* output,
                                       slate_backward_fn backward);

// Run reverse-mode autodiff starting from `loss`. The loss tensor must be a
// scalar (numel == 1) with a grad buffer that the caller has set to 1.0.
slate_status_t slate_graph_backward(slate_graph_ctx_t* ctx,
                                    slate_tensor_t* loss);

#ifdef __cplusplus
}
#endif

#endif // SLATE_AUTOGRAD_H
