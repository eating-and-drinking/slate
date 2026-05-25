// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// checkpoint.h — gradient checkpointing.
//
// `slate_op_checkpoint(ctx, x, fn, ud)` runs `fn(ctx', x)` in a sub-context
// that does NOT record intermediate activations. On backward, the input x
// is replayed through `fn` in a fresh sub-context (this time WITH activation
// recording), and the gradient flows backward through that ephemeral graph.
//
// Memory: between forward and backward, only x and fn's final output are
// retained. Activations inside fn are discarded after forward and
// rematerialized only during the backward pass.

#ifndef SLATE_CHECKPOINT_H
#define SLATE_CHECKPOINT_H

#include "slate/types.h"
#include "slate/autograd.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef slate_tensor_t* (*slate_checkpoint_fn)(slate_graph_ctx_t* ctx,
                                                 slate_tensor_t* x,
                                                 void* user_data);

slate_tensor_t* slate_op_checkpoint(slate_graph_ctx_t* ctx,
                                     slate_tensor_t* x,
                                     slate_checkpoint_fn fn,
                                     void* user_data);

#ifdef __cplusplus
}
#endif

#endif
