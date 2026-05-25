// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// transformer.h — modules for decoder-only transformers (M2).
// Single-head attention; multi-head and RoPE are deferred to M3+.

#ifndef SLATE_TRANSFORMER_H
#define SLATE_TRANSFORMER_H

#include "slate/types.h"
#include "slate/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Embedding: integer lookup. weight [V, D]; forward(indices [..]) -> [.., D].
slate_module_t* slate_module_embedding_new(slate_arena_t* params,
                                            int vocab_size, int d_model,
                                            uint64_t seed);

// RMSNorm over the last dim with learnable weight [d_model].
slate_module_t* slate_module_rmsnorm_new(slate_arena_t* params,
                                          int d_model, float eps);

// Single-head causal self-attention: input [B, T, D] -> output [B, T, D].
slate_module_t* slate_module_attention_new(slate_arena_t* params,
                                            int d_model,
                                            uint64_t seed);

// FFN with SwiGLU: hidden_dim ~ 4*d_model (rounded).
slate_module_t* slate_module_ffn_new(slate_arena_t* params,
                                      int d_model, int hidden_dim,
                                      uint64_t seed);

// One transformer block: x = x + attn(rms_norm(x)); x = x + ffn(rms_norm(x)).
slate_module_t* slate_module_transformer_block_new(slate_arena_t* params,
                                                    int d_model, int ffn_hidden,
                                                    float norm_eps,
                                                    uint64_t seed);

// Full GPT-style decoder LM.
//   forward(token_ids [B, T]) -> logits [B, T, vocab_size]
slate_module_t* slate_module_causal_lm_new(slate_arena_t* params,
                                            int vocab_size, int max_seq_len,
                                            int d_model, int n_layers,
                                            int ffn_hidden, float norm_eps,
                                            uint64_t seed);

slate_module_t* slate_module_mh_attention_new(slate_arena_t* params,
                                               int d_model, int n_heads,
                                               uint64_t seed);

slate_module_t* slate_module_lora_new(slate_arena_t* params,
                                       int in_features, int out_features,
                                       int rank, float alpha,
                                       const float* base_weight,
                                       uint64_t seed);

typedef struct slate_gguf slate_gguf_t;
slate_module_t* slate_module_quantized_lora_new(slate_arena_t* params,
                                                 slate_gguf_t* gguf,
                                                 const char* tensor_name,
                                                 int rank, float alpha,
                                                 uint64_t seed);

// Forward declaration used by the linear3d op (referenced by attention/ffn).
slate_tensor_t* slate_op_linear3d(slate_graph_ctx_t* ctx,
                                   slate_tensor_t* x, slate_tensor_t* W);

#ifdef __cplusplus
}
#endif

#endif
