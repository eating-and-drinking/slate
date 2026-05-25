// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// stream.h — sub-module weight streaming for training models that exceed RAM.
//
// A StreamUnit is one piece of the model that can be loaded into RAM,
// participate in a forward (and optionally backward) pass, and be evicted.
// Sub-module granularity is the default: each transformer block contributes
// two units, one for attention and one for FFN.
//
// IMPLEMENTATION STATUS: M5. The header is stable; src/stream/ is empty.

#ifndef SLATE_STREAM_H
#define SLATE_STREAM_H

#include "slate/types.h"
#include "slate/tensor.h"
#include "slate/autograd.h"

#ifdef __cplusplus
extern "C" {
#endif

// Runtime mode determines how StreamUnits behave on load and during ops.
//
//   INFERENCE        all weights resident, KV cache active, no grads recorded
//   TRAINING         weights streamed in/out per unit, grads allocated,
//                    selective checkpoint on, KV cache off
//   TEACHER_SCORING  weights streamed, no grads, no KV cache (used to
//                    compute teacher top-k logits over a fixed sequence)
typedef enum slate_runtime_mode {
    SLATE_MODE_INFERENCE       = 0,
    SLATE_MODE_TRAINING        = 1,
    SLATE_MODE_TEACHER_SCORING = 2,
} slate_runtime_mode_t;

typedef struct slate_kv_cache slate_kv_cache_t;
typedef struct slate_act_cache slate_act_cache_t;  // saved-for-backward
typedef struct slate_stream_unit slate_stream_unit_t;

// Streaming policy: where weights live when not loaded, and the selective
// checkpoint strategy.
typedef struct slate_stream_policy {
    const char* root_path;            // directory containing block_*.bin files
    slate_runtime_mode_t mode;
    bool prefetch_next;               // overlap I/O with compute
    int ring_buffer_units;            // typically 2 (current + prefetch) or 3
    // Selective checkpoint policy: bitmask of activations to keep.
    enum {
        SLATE_CKPT_NONE          = 0,
        SLATE_CKPT_ATTN_OUTPUT   = 1 << 0,
        SLATE_CKPT_FFN_INTERMED  = 1 << 1,  // expensive, usually off
        SLATE_CKPT_BLOCK_INPUT   = 1 << 2,  // always on
    } selective_keep_mask;
} slate_stream_policy_t;

// The StreamUnit interface. Concrete units (Attention sub-module, FFN
// sub-module, or whatever else gets streamed) implement this vtable.
struct slate_stream_unit {
    const char* name;                 // for logs and the on-disk filename

    slate_status_t (*load)  (slate_stream_unit_t* self, slate_arena_t* weights);
    slate_tensor_t* (*forward)(slate_stream_unit_t* self,
                               slate_graph_ctx_t* ctx,
                               slate_tensor_t* in,
                               slate_kv_cache_t* kv);
    void (*save_for_backward)(slate_stream_unit_t* self, slate_act_cache_t* cache);
    slate_tensor_t* (*backward)(slate_stream_unit_t* self,
                                slate_graph_ctx_t* ctx,
                                slate_tensor_t* d_out,
                                slate_act_cache_t* cache);
    void (*evict)(slate_stream_unit_t* self);

    void* user_data;
};

// =============================================================================
// Runtime orchestrator.
// =============================================================================
//
// Drives a sequence of StreamUnits according to the policy, including
// prefetching, eviction, selective checkpointing, and the
// generation-vs-training mode switch.

typedef struct slate_stream_runtime slate_stream_runtime_t;

slate_stream_runtime_t* slate_stream_runtime_new(slate_stream_unit_t** units,
                                                  int n_units,
                                                  const slate_stream_policy_t* policy);

void slate_stream_runtime_destroy(slate_stream_runtime_t* rt);

void slate_stream_runtime_set_mode(slate_stream_runtime_t* rt,
                                    slate_runtime_mode_t mode);

// Run a forward pass through the entire pipeline. Returns the final output.
slate_tensor_t* slate_stream_runtime_forward(slate_stream_runtime_t* rt,
                                              slate_graph_ctx_t* ctx,
                                              slate_tensor_t* input,
                                              slate_kv_cache_t* kv);

// Run a backward pass. Must be called after forward with matching mode.
slate_status_t slate_stream_runtime_backward(slate_stream_runtime_t* rt,
                                              slate_graph_ctx_t* ctx,
                                              slate_tensor_t* loss);

#ifdef __cplusplus
}
#endif

#endif // SLATE_STREAM_H
