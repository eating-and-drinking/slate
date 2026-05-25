// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// adapter.h — LoRA adapters and the AdapterManager for nightly training.
//
// An adapter is a small set of trainable parameters that overlays a frozen
// base model. LoRA decomposes each weight update ΔW into a low-rank product
// A · B where rank(A·B) ≪ rank(W). This shrinks training memory from
// O(base_size) to O(rank · sum_dims) — typically by 100×.
//
// IMPLEMENTATION STATUS: M5.

#ifndef SLATE_ADAPTER_H
#define SLATE_ADAPTER_H

#include "slate/types.h"
#include "slate/module.h"
#include "slate/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// A single LoRA adapter wrapping one base weight matrix.
//
//   y = x · (W_base + α/r · A · B)
//
// where A is [in, r], B is [r, out], α is a scaling factor.
typedef struct slate_lora_adapter slate_lora_adapter_t;

slate_lora_adapter_t* slate_lora_adapter_new(slate_arena_t* param_arena,
                                              int in_features,
                                              int out_features,
                                              int rank,
                                              float alpha,
                                              uint64_t init_seed);

slate_tensor_t* slate_lora_adapter_forward(slate_lora_adapter_t* a,
                                            slate_graph_ctx_t* ctx,
                                            slate_tensor_t* x,
                                            slate_tensor_t* base_y);

void slate_lora_adapter_register_params(slate_lora_adapter_t* a,
                                         slate_param_set_t* ps);

// =============================================================================
// AdapterManager: nightly-training adapter lifecycle.
// =============================================================================
//
// Manages the file-system layout:
//
//   $root/adapters/current.lora        - active for inference
//   $root/adapters/training.lora.tmp   - tonight's in-progress
//   $root/adapters/archive/<date>.lora - historical versions
//
// All transitions are atomic (write to tmp, fsync, rename).

typedef struct slate_adapter_manager slate_adapter_manager_t;

slate_adapter_manager_t* slate_adapter_manager_new(const char* root_path);
void slate_adapter_manager_destroy(slate_adapter_manager_t* mgr);

// Save tonight's training adapter as the new candidate.
slate_status_t slate_adapter_manager_save_training(slate_adapter_manager_t* mgr,
                                                    const slate_param_set_t* params);

// Compare candidate vs current using the eval gate. If candidate wins,
// atomically promote it; otherwise discard. Returns SLATE_OK and sets
// `*promoted` to true/false.
slate_status_t slate_adapter_manager_eval_and_promote(slate_adapter_manager_t* mgr,
                                                      bool* promoted);

// Roll back to a historical adapter by date string ("YYYY-MM-DD").
slate_status_t slate_adapter_manager_rollback(slate_adapter_manager_t* mgr,
                                               const char* date);

#ifdef __cplusplus
}
#endif

#endif // SLATE_ADAPTER_H
