# Roadmap: getting slate to actually run LLaMA-class GGUF models

This document lays out, **honestly**, what's still missing between
slate's current state and the point where you can `slate_inference_server`
on a `llama-7b-Q4_K_M.gguf` file downloaded from Hugging Face and serve
real completions. It's split into "what works today" and "what's
genuinely needed" so the gap can't be hand-waved away.

## What works today (end of L8.6)

- **Pure CPU training stack** — autograd, optimizers (SGD/AdamW/
  Adafactor/Muon), LoRA, gradient checkpointing.
- **KV-cached single-token decode** at O(L·D) per token, bit-identical
  to the training-side forward (`test_infer`).
- **Continuous batching**: B-wide batched decode with linear projections
  fused as M=B GEMMs, ~10× tok/s at B=16 vs B=1 (`bench_batch_throughput`).
- **HTTP/1.1 server** with SSE streaming, multi-key auth + per-key
  token-bucket rate limiting, Prometheus metrics, structured JSON
  logs, graceful SIGTERM, and an in-process micro-batching scheduler
  that brings the batched throughput to real concurrent load
  (1.79× end-to-end measured on 8 clients).
- **Quantized inference**: Q8_0 and Q4_K_M dequant + dot kernels with
  AVX2. Q4_K_M reaches 10.59 GFLOP/s — **faster than Q8_0** while
  using half the memory.
- **GGUF reader** can now parse Q4_K_M tensors (`test_gguf_q4k`):
  end-to-end load → dequant → matvec, bit-identical to the Python
  reference reconstruction.

## What's genuinely missing for LLaMA-7B Q4_K_M

These are *not* small. None of them have been started yet.

### 1. ~~RoPE (Rotary Position Embedding)~~ — **DONE (L9.2)**

Forward + autograd backward in src/ops/rope.c, AVX2-friendly scalar (no SIMD yet but the loop is amenable). Bit-precise vs numpy reference (Linf = 2.4e-7); inverse round-trip Linf = 4.8e-7. See tests/test_rope.c.

### 1b. RoPE (original section, for context)

LLaMA does not use additive position embeddings. It rotates the
per-head Q and K vectors by a per-position frequency table *inside*
the attention computation:

```
q', k' = RoPE(q, k, position, theta_base=10000)
attention(q', k', v)  // same as before
```

Slate's current `slate_module_attention` and the inference engine's
`attention_step` both consume Q and K as-is, with pos_emb added at
the embedding layer.  Need:

- A `slate_op_rope` op that rotates `[B, T, n_heads, head_dim]` Q and
  K tensors by the GGML standard formula (interleaved or non-interleaved
  — depends on the model);
- Apply it after Q and K projections, before the score matmul;
- In the **inference engine**, store K in cache *post-RoPE* so attention
  remains scale-correct.

Effort: ~300 lines C + tests vs a reference Python RoPE implementation.

### 2. Grouped-Query Attention (GQA)

LLaMA-2 7B uses MHA (Q heads = K/V heads). But LLaMA-2 70B, LLaMA-3
and Mistral all use GQA where K/V heads < Q heads (typically 8 K/V
for 32 Q in 7B-class models, but check the model config).

Slate's MHA module hard-codes `n_kv_heads == n_q_heads`. Need:
- A `n_kv_heads` parameter,
- K/V matrices of shape `[d_model, n_kv_heads × head_dim]` (smaller),
- Q from each head reads from the K/V head it's grouped to during
  attention (head_idx_kv = head_idx_q // (n_q_heads / n_kv_heads)).

Effort: ~150 lines C + tests.

### 3. SwiGLU FFN exists but check intermediate-dim convention

LLaMA's FFN hidden dim is `int(2/3 * 4 * d_model)` rounded up to the
nearest 256-multiple (e.g. d_model=4096 → 11008, not 16384). Slate's
`slate_module_ffn_new` takes hidden_dim as a parameter — fine, just
need to pass the right number from the GGUF metadata.

Effort: zero code, just metadata parsing in the GGUF→slate loader.

### 4. ~~Weight-naming convention~~ — **DONE (L9.3)**

`slate_llama_open(gguf)` resolves all standard llama.cpp-named tensors and `llama.*` metadata into a `slate_llama_t` view. Tied-output is auto-detected. GGUF reader exposes `slate_gguf_get_u32/f32/str` for arbitrary metadata keys. See tests/test_llama_load.c.

### 4b. Weight-naming convention (original section)

GGUF stores LLaMA weights as:
```
token_embd.weight                 # [V, D]
blk.0.attn_norm.weight            # [D]
blk.0.attn_q.weight               # [D, D]
blk.0.attn_k.weight               # [D, n_kv_heads*head_dim]
blk.0.attn_v.weight               # [D, n_kv_heads*head_dim]
blk.0.attn_output.weight          # [D, D]
blk.0.ffn_norm.weight             # [D]
blk.0.ffn_gate.weight             # [hidden, D]
blk.0.ffn_up.weight               # [hidden, D]
blk.0.ffn_down.weight             # [D, hidden]
output_norm.weight                # [D]
output.weight                     # [D, V]  -- possibly tied with token_embd
```

Slate's `slate_infer_engine_new` extracts weights by **walking the
param-set in slate's known order** (`ps[0]=tok_emb, ps[1]=pos_emb, ...`).
That won't match a GGUF that was produced by llama.cpp.  Need:

- A `slate_llama_model_t` that holds named pointers into the GGUF data
  (mmap pages, no copies);
- An `slate_infer_engine_new_from_llama(slate_llama_model_t*)`
  constructor;
- All the existing inference functions (prefill, decode_step,
  batch_step) parameterised over the model type.

Effort: ~250 lines C.

### 5. Tokenizer

LLaMA uses SentencePiece BPE with a specific vocab and byte-pair table
stored in the GGUF. Slate has its own BPE tokenizer (`src/tokenizer/bpe.c`)
that's incompatible.  Need either:
- A SentencePiece-style loader that reads the GGUF tokenizer metadata
  and rebuilds the merge table; or
- Cheat: accept already-tokenised prompts (integer arrays) and skip
  decoding back to text. Acceptable for benchmarks, not for production.

Effort: full tokenizer ~400 lines; cheat mode ~30 lines.

### 6. Tied output projection

Most LLaMAs tie `output.weight == token_embd.weight` (one matrix, two
roles). Slate's `slate_module_causal_lm` keeps them separate. Need a
`tied=true` flag in the LLaMA model wrapper.

Effort: ~20 lines.

### 7. Performance gap to llama.cpp

llama.cpp at LLaMA-7B Q4_K_M on a modern x86 CPU does ~10–15 tok/s
single-thread, ~30–40 tok/s with 8 threads. Slate's expected number,
extrapolating from our 10.59 GFLOP/s matvec:

  6.5B params × 2 ops × 1 step = 13 GFLOP per token
  ÷ 10.59 GFLOP/s ≈ **1.2 seconds per token (single-thread)**

That's ~10× slower than llama.cpp. The headroom:
- We don't have AVX-512 (llama.cpp does);
- We don't yet thread the Q4_K matvec across CPU cores;
- Our prefill is single-token (llama.cpp batches the whole prompt);
- We haven't optimised attention with Flash-Attention-style tiling.

So even with everything above done, expect "works but slow" on
LLaMA-7B until items 1.X (multi-thread Q4_K) and 1.X (Flash Attention)
land. The order I'd suggest:

1. RoPE                 — unblocks any decoder-only LLaMA-class model
2. GGUF→slate weight mapping — actually load a real .gguf
3. GQA                  — needed for everything ≥ LLaMA-2 13B
4. Tokenizer (cheat mode is fine for benchmarks)
5. Multi-thread Q4_K    — easy throughput win
6. Flash-Attention tile — closes most of the remaining gap

Total: probably 1.5–2 more sessions of focused work before the first
real LLaMA-7B token comes out of slate. Honest estimate.

### What this milestone (L8.7) actually delivered

- GGUF reader recognises Q4_K_M (type 12).
- An end-to-end Q4_K_M load + dequant + dot + matvec test that's
  bit-identical to the Python reference.
- The 2.4 KB synthetic Q4_K_M GGUF fixture (`tools/make_q4k_gguf.py`).
- This document — so the gap to real LLaMA-7B is no longer
  hand-wavy.
