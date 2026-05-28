// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// infer.h — CPU inference engine with KV cache.
//
// Slate's training path goes through the autograd graph; that's correct
// for training but expensive at decode time, because each next-token
// step re-computes K and V for the entire prefix.  This header carves
// out an inference-only fast path that:
//
//   * pre-allocates per-layer K and V caches sized to a maximum
//     conversation length;
//   * exposes a "prefill" call (process the prompt) and a "decode"
//     call (one step per new token); decode is O(L·D) per token where
//     L is the cache length so far, not O(L²·D);
//   * does NOT touch the autograd graph — there are no node arenas,
//     no gradient buffers, no backward, no scratch tensor objects.
//     Forward operations work directly on float* buffers and the same
//     packed-panel AVX2 matmul that powers training.
//
// The engine consumes a trained `slate_module_causal_lm` model: it
// extracts the weight tensors by walking the parameter set the module
// registers, in the deterministic order documented below.
//
// Layout (single-head attention causal LM; same as
// slate_module_causal_lm_new produces):
//   ps[0]                    : tok_emb     [V, D]
//   ps[1]                    : pos_emb     [max_seq, D]
//   per layer i in 0..L-1, base = 2 + i*9:
//     ps[base+0]             : norm1_w     [D]
//     ps[base+1]             : Wq          [D, D]
//     ps[base+2]             : Wk          [D, D]
//     ps[base+3]             : Wv          [D, D]
//     ps[base+4]             : Wo          [D, D]
//     ps[base+5]             : norm2_w     [D]
//     ps[base+6]             : Wg          [D, H]    ffn gate
//     ps[base+7]             : Wu          [D, H]    ffn up
//     ps[base+8]             : Wd          [H, D]    ffn down
//   ps[2 + L*9 + 0]          : final_norm_w[D]
//   ps[2 + L*9 + 1]          : lm_head     [D, V]
//
// Thread-safety: the engine is read-only after construction and can be
// shared across threads.  A `slate_infer_session_t` is per-conversation
// state (KV cache + position) and is NOT safe to share — each request
// in a server should get its own session.

#ifndef SLATE_INFER_H
#define SLATE_INFER_H

#include "slate/types.h"
#include "slate/module.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_infer_engine  slate_infer_engine_t;
typedef struct slate_infer_session slate_infer_session_t;

// Build an inference engine from a trained slate_module_causal_lm.
// The model must remain alive for the lifetime of the engine
// (weights are referenced, not copied).
//
//   model      : a slate_module_causal_lm_new(...)-produced module
//   n_layers   : number of transformer blocks in the model
//   d_model    : hidden size
//   vocab      : vocabulary size
//   ffn_hidden : FFN inner dim
//   max_seq    : maximum sequence length supported by sessions opened
//                from this engine (sets KV cache size)
//
// Returns NULL on failure (NULL model, allocation error, mismatched
// parameter count).
slate_infer_engine_t* slate_infer_engine_new(slate_module_t* model,
                                              int n_layers,
                                              int d_model,
                                              int vocab,
                                              int ffn_hidden,
                                              int max_seq);

// Like slate_infer_engine_new but with an explicit compute backend.
// Passing NULL uses slate_backend_default() (CPU until a GPU backend is
// linked in).  When wiring up a real CUDA / Metal backend, this is the
// constructor to use — the engine's hot path routes every primitive
// (matvec, linear_batch, rmsnorm_row, silu_mul, embed_lookup,
// attention_step, add_inplace) through the backend's vtable, and
// session-side scratch (KV cache + activations) is allocated via
// backend->alloc / freed via backend->release.
struct slate_backend;   // forward decl, defined in slate/backend.h
slate_infer_engine_t* slate_infer_engine_new_ex(slate_module_t* model,
                                                 int n_layers,
                                                 int d_model,
                                                 int vocab,
                                                 int ffn_hidden,
                                                 int max_seq,
                                                 const struct slate_backend* backend);

void slate_infer_engine_free(slate_infer_engine_t* eng);

// Vocab size of the model wrapped by this engine.  Needed by callers
// that want to size logits buffers / clamp samples to valid token ids.
int slate_infer_engine_vocab(const slate_infer_engine_t* eng);

// Max sequence length supported by sessions opened from this engine.
int slate_infer_engine_max_seq(const slate_infer_engine_t* eng);

// Open a per-conversation session.  Allocates KV cache + activation
// scratch.  Memory cost is roughly 2 * n_layers * max_seq * d_model *
// sizeof(float).  Returns NULL on allocation failure.
slate_infer_session_t* slate_infer_session_new(slate_infer_engine_t* eng);

void slate_infer_session_free(slate_infer_session_t* sess);

// Reset the session for a new conversation (zero position, no
// deallocation).
void slate_infer_session_reset(slate_infer_session_t* sess);

// Number of tokens currently in the KV cache for this session.
int slate_infer_session_position(const slate_infer_session_t* sess);

// Process a prompt of `n_tokens` tokens.  This is equivalent to N
// successive decode steps but avoids the per-step softmax-allocation
// overhead; the result is the same as if you'd called decode_step N
// times.  Returns the logits at the *last* prompt position
// (`out_logits` must have `vocab` floats).
//
// Errors:
//   * returns < 0 if the cumulative position would exceed max_seq;
//   * returns < 0 on NULL session/tokens.
int slate_infer_prefill(slate_infer_session_t* sess,
                         const int32_t* tokens, int n_tokens,
                         float* out_logits);

// Process one new token: appends K, V into the cache and produces
// logits for the next-token distribution.  out_logits must have
// `vocab` floats.  Returns 0 on success, < 0 on error.
int slate_infer_decode_step(slate_infer_session_t* sess,
                             int32_t token,
                             float* out_logits);

// Max sequence length supported by sessions opened from this engine.
int slate_infer_engine_max_seq(const slate_infer_engine_t* eng);



// ---------------------------------------------------------------------------
// Batched inference: continuous batching across N concurrent sessions.
// ---------------------------------------------------------------------------
//
// The single-session decode loop in slate_infer_decode_step issues all of
// its linear projections (Wq/Wk/Wv/Wo/Wg/Wu/Wd + lm_head) as M=1 GEMMs,
// which is the worst-case shape for our packed-panel kernel — most of
// the per-call overhead (panel packing, tile setup) is amortised against
// just one row of output.  A `slate_infer_batch_t` lets the caller stack
// B in-flight sessions and run one M=B GEMM per layer instead of B
// separate M=1 GEMMs.  Empirically this gives 2–4x throughput once B≥4.
//
// Per-session attention is still computed independently because each
// session has its own KV cache length — but attention is the cheap part
// for small models, and the FFN GEMMs (which dominate) batch cleanly.
//
// Sessions in the batch can have *different* current positions; the
// engine looks up `pos_emb[sess->position]` and does the full cache-aware
// attention per session.  This is true continuous-batching semantics
// (each request progresses at its own pace) — the only requirement is
// that B does not exceed the batch object's `max_batch`.

typedef struct slate_infer_batch slate_infer_batch_t;

slate_infer_batch_t* slate_infer_batch_new(slate_infer_engine_t* eng,
                                            int max_batch);

void slate_infer_batch_free(slate_infer_batch_t* batch);

// Advance `n_sessions` sessions by one token each.  Token i is appended
// to sessions[i]'s cache; logits for sessions[i] are written to
// out_logits[i*vocab .. (i+1)*vocab).
//
// All sessions must have been opened from the same engine.  n_sessions
// must be in [1, max_batch].
//
// Returns 0 on success, < 0 on error.
int slate_infer_batch_step(slate_infer_batch_t* batch,
                            slate_infer_session_t** sessions,
                            int n_sessions,
                            const int32_t* tokens,
                            float* out_logits);

#ifdef __cplusplus
}
#endif

#endif // SLATE_INFER_H
