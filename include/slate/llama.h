// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// llama.h — LLaMA-architecture GGUF wrapper.
//
// llama.cpp stores LLaMA / Mistral / Qwen-class models in GGUF v3 with
// a specific tensor-naming convention and a set of `llama.*` metadata
// keys.  This wrapper parses that schema so slate can load real
// llama-7b-Q4_K_M.gguf-style files without rolling its own format.
//
// Conventions parsed:
//   * Tensor names: `token_embd.weight`, `output_norm.weight`,
//     `output.weight` (may be absent if tied to `token_embd.weight`),
//     and per-layer:  `blk.N.attn_norm.weight`, `blk.N.attn_q.weight`,
//     `blk.N.attn_k.weight`, `blk.N.attn_v.weight`,
//     `blk.N.attn_output.weight`, `blk.N.ffn_norm.weight`,
//     `blk.N.ffn_gate.weight`, `blk.N.ffn_up.weight`,
//     `blk.N.ffn_down.weight`.
//   * Metadata keys (under the "llama." namespace):
//     - llama.block_count            (u32) → n_layers
//     - llama.embedding_length       (u32) → d_model
//     - llama.feed_forward_length    (u32) → ffn_hidden
//     - llama.attention.head_count   (u32) → n_heads
//     - llama.attention.head_count_kv(u32) → n_kv_heads (GQA; falls
//                                            back to n_heads if absent)
//     - llama.context_length         (u32) → max_seq
//     - llama.rope.freq_base         (f32) → theta_base (default 10000)
//     - llama.attention.layer_norm_rms_epsilon (f32) → rms_eps
//     - llama.vocab_size             (u32) → vocab (also derivable from
//                                            token_embd shape)
//
// This wrapper does NOT yet run inference — it just resolves the
// weight pointers and config from disk.  Pair with a future
// slate_infer_engine_new_from_llama() (next milestone) for full
// inference.  The current value is: any LLaMA-format GGUF can be
// safely opened, its config introspected, and its weight tensors
// accessed by named accessor.

#ifndef SLATE_LLAMA_H
#define SLATE_LLAMA_H

#include "slate/types.h"
#include "slate/gguf.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_llama        slate_llama_t;
typedef struct slate_llama_layer  slate_llama_layer_t;

typedef struct slate_llama_config {
    int   n_layers;
    int   d_model;
    int   ffn_hidden;
    int   n_heads;
    int   n_kv_heads;      // == n_heads for MHA (LLaMA-2-7B); < for GQA
    int   head_dim;        // d_model / n_heads
    int   max_seq;         // llama.context_length
    int   vocab;
    float theta_base;      // RoPE freq base (default 10000.0)
    float rms_eps;         // RMSNorm epsilon (default 1e-5)
    int   tied_output;     // 1 if output.weight == token_embd.weight
} slate_llama_config_t;

// Per-block weight pointers.  All point INTO the underlying GGUF's
// mmap'd data (no copies).  All are owned by the gguf.
struct slate_llama_layer {
    const void* attn_norm;     // [d_model]              f32 (RMSNorm gain)
    const void* attn_q;        // [d_model, d_model]     dtype = config quant
    const void* attn_k;        // [d_model, n_kv_heads*head_dim]
    const void* attn_v;        // [d_model, n_kv_heads*head_dim]
    const void* attn_output;   // [d_model, d_model]
    const void* ffn_norm;      // [d_model]              f32
    const void* ffn_gate;      // [d_model, ffn_hidden]
    const void* ffn_up;        // [d_model, ffn_hidden]
    const void* ffn_down;      // [ffn_hidden, d_model]
    int attn_q_dtype;          // slate_dtype_t for each block (usually same for all)
    int attn_k_dtype;
    int attn_v_dtype;
    int attn_output_dtype;
    int ffn_gate_dtype;
    int ffn_up_dtype;
    int ffn_down_dtype;
};

// Parse a LLaMA-format GGUF.  Returns NULL on missing metadata /
// missing required tensors / unsupported variant.
//
// The caller retains ownership of `gguf` — the returned slate_llama_t
// borrows pointers into it and must be freed BEFORE the gguf is closed.
slate_llama_t* slate_llama_open(slate_gguf_t* gguf);

void slate_llama_free(slate_llama_t* m);

// Accessors
const slate_llama_config_t* slate_llama_config(const slate_llama_t* m);
const void*                 slate_llama_token_embd(const slate_llama_t* m);
int                         slate_llama_token_embd_dtype(const slate_llama_t* m);
const void*                 slate_llama_output_norm(const slate_llama_t* m);  // f32
const void*                 slate_llama_output(const slate_llama_t* m);
int                         slate_llama_output_dtype(const slate_llama_t* m);
const slate_llama_layer_t*  slate_llama_layer(const slate_llama_t* m, int layer);

// Dump config and a short tensor-presence summary to stdout.
void slate_llama_dump(const slate_llama_t* m);

// -----------------------------------------------------------------------------
// Inference: session-based, KV-cached, multi-head with RoPE.
// -----------------------------------------------------------------------------
//
// LLaMA-style decode-only inference.  Currently supports:
//   * MHA (n_kv_heads == n_heads).  GQA is on the roadmap.
//   * RoPE on Q and K (theta_base from the model's metadata).
//   * Tied output projection (auto-detected by slate_llama_open).
//   * f32 weights only (Q4_K_M support is roadmap'd separately).
//
// Memory: KV cache = 2 * n_layers * max_seq * d_model * sizeof(float).
// For LLaMA-7B with max_seq=2048: 2 * 32 * 2048 * 4096 * 4 = 2 GB per
// session.  Production code would want paged KV cache to drop that
// to working-set size, but the API surface is the same.

typedef struct slate_llama_session slate_llama_session_t;

slate_llama_session_t* slate_llama_session_new(const slate_llama_t* model);
void                   slate_llama_session_free(slate_llama_session_t* sess);
int                    slate_llama_session_position(const slate_llama_session_t* sess);

// Process `n_tokens` prompt tokens.  Returns the logits at the LAST
// prompt position into `out_logits[vocab]`.  Returns 0 on success,
// < 0 on error (cache overflow, NULL inputs).
int slate_llama_prefill(slate_llama_session_t* sess,
                         const int32_t* tokens, int n_tokens,
                         float* out_logits);

// Process one new token, return next-position logits.
int slate_llama_decode_step(slate_llama_session_t* sess,
                             int32_t token, float* out_logits);

#ifdef __cplusplus
}
#endif

#endif // SLATE_LLAMA_H
