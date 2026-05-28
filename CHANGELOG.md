# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Grouped-Query Attention (GQA) (`L9.6`)

slate's LLaMA inference path now supports GQA — fewer K/V heads than
Q heads.  Required by **LLaMA-3-8B** (32 Q / 8 K/V), **LLaMA-3-70B**
(64/8), **Mistral-7B** (32/8), and every modern model larger than
LLaMA-2-7B.  MHA continues to work as the trivial case (`group=1`).

- **`attention_multihead`** (`src/loader/llama_infer.c`) generalised:
  each Q head `hq` reads from K/V head `hq / group`, where `group =
  n_q_heads / n_kv_heads`.  KV cache shape is already correctly sized
  `[L, n_kv_heads * head_dim]` (the L9.3 loader read `n_kv_heads`
  from `llama.attention.head_count_kv` metadata, so this just needed
  the attention loop to honour it).

- **MHA bit-identical**: existing `test_llama_infer` still passes
  with `L_inf = 5.215e-8` (same as before this milestone).  GQA when
  `n_kv_heads == n_heads` is the same code path as MHA.

- **`tools/make_tiny_llama_gqa_gguf.py`**: synthetic GQA-shaped
  LLaMA GGUF — `n_q_heads = 8`, `n_kv_heads = 2` (`group = 4`,
  matching LLaMA-3-8B's pattern).  Reference numpy forward implements
  the same grouped attention.

- **Test** (`tests/test_llama_infer_gqa.c`): verifies slate's GQA
  forward matches the numpy reference at **L_inf = 8.94e-8,
  L2 = 3.11e-7** — bit-precise within fp32.

- **`ROADMAP_LLAMA.md`** updated: item 3 (GQA) crossed off.  The
  last remaining items are:
    - SentencePiece tokenizer (or cheat mode for benchmarks)
    - Multi-thread Q4_K matvec (performance, not correctness)
    - Flash Attention 2 (performance)

Total test count: 33/33 passing.

### Added — Q4_K LLaMA inference (`L9.5`)

slate's LLaMA inference now consumes **Q4_K_M quantised weights**
directly — no f32 materialisation, no dequant-then-matmul round trip.
This is the configuration real `llama-7b-Q4_K_M.gguf` files ship in
(3.8 GB for a 7B model vs 26 GB for f32), and the one needed for
slate to actually serve those models on commodity hardware.

- **`slate_backend_t::matvec_q4k`** (`include/slate/backend.h`,
  `src/backend/backend_cpu.c`): new primitive,
  `void matvec_q4k(y, A_Q4K, x, M, K)`.  CPU backend wraps the
  existing `slate_q4_k_matvec` (10.59 GFLOP/s on AVX2). CUDA/Metal
  stubs left as NULL with documentation.

- **`src/loader/llama_infer.c`**: per-weight dtype dispatch via a
  small `mv_dispatch()` helper.  Each linear projection
  (Wq, Wk, Wv, Wo, Wg, Wu, Wd, lm_head) picks `matvec_q4k` when the
  loaded weight is `SLATE_DTYPE_Q4_K`, falls back to `matvec` when
  it's `SLATE_DTYPE_F32`.  This means mixed-precision GGUFs (e.g.
  Q4_K linear weights + f32 norms + f32 embedding) work transparently.

- **`tools/make_tiny_llama_q4k_gguf.py`**: emits a synthetic LLaMA
  GGUF where every projection weight is Q4_K_M (norms and embedding
  stay f32 — matches real `llama-7b-Q4_K_M.gguf` layout).  Model
  dims chosen so every K is a multiple of 256: V=64, D=256,
  n_heads=4, head_dim=64, FFN=512.  875 KB, 21 tensors, 11 KV
  metadata.

- **Test** (`tests/test_llama_infer_q4k.c`): opens the Q4_K GGUF,
  verifies `attn_q_dtype == ffn_gate_dtype == 18` (Q4_K), runs
  prefill on a 5-token prompt, compares to numpy reference that
  uses the *dequantised* weights. **Linf = 4.17e-7, L2 = 1.43e-6** —
  bit-precise within fp32 (the Q4_K accumulation error is below
  fp32's mantissa resolution at this model size).

- **`ROADMAP_LLAMA.md`** updated: the remaining gap to a real
  `llama-7b-Q4_K_M.gguf` is now:
  1. Accept int32 prompts directly (cheat mode, ~30 lines) — needed
     to bypass tokenization until SentencePiece lands
  2. Larger `max_seq` (config-only, no code)
  3. GQA for ≥13B class — slate-7B works as-is with MHA
  4. Optionally multi-thread Q4_K matvec for higher tok/s

Total test count: 32/32 passing.

### Added — LLaMA inference path (`L9.4`)

slate can now actually run a LLaMA-architecture model from a GGUF file.
This closes the gap to "load real LLaMA-7B-Q4_K_M.gguf and stream
tokens out" (modulo Q4_K weight unpacking inside matvec + a
tokenizer, both of which are downstream of this).

- **`include/slate/llama.h` (extended)** + **`src/loader/llama_infer.c`**:
  - `slate_llama_session_t` — per-conversation state (per-layer KV
    caches + position + scratch), allocated from the model's vocab/D/
    n_heads/ffn config.
  - `slate_llama_prefill(sess, tokens, n, out_logits)` — N successive
    decode steps; returns last-position logits.
  - `slate_llama_decode_step(sess, token, out_logits)` — single
    decode advance; appends K/V into cache, returns logits.
  - Architecture: multi-head causal self-attention with **RoPE
    applied to Q and K post-projection** (per llama.cpp), SwiGLU FFN,
    RMSNorm pre-attn + pre-FFN, optional tied output projection.
  - MHA only for now (n_kv_heads must equal n_heads). GQA shape is
    plumbed through the layer struct but the attention loop assumes
    MHA — GQA generalisation is roadmap'd separately (~150 lines
    diff).

- **Bug fix (loader convention)**: `tools/make_tiny_llama_gguf.py`
  and `tools/make_llama_ref.py` were originally using numpy
  `[in, out]` shape for linear weights. The real llama.cpp GGUF
  convention is **`[out, in]`** (output rows first; matvec consumes
  the input vector). Both files were corrected, and `decode_one`
  in `llama_infer.c` now uses `backend->matvec(y, W, x, M=out, K=in)`
  instead of `linear_batch` for all linear projections. Without
  this, slate's output drifts ~0.4 Linf from the reference; with it,
  drift is `5.2e-8` (fp32 precision floor).

- **Test** (`tests/test_llama_infer.c`): opens the synthetic
  LLaMA GGUF, loads + sessions, runs prefill on a 5-token prompt,
  compares last-position logits to the numpy reference dumped by
  `tools/make_llama_ref.py`. **Linf = 5.215e-08, L2 = 1.649e-07** —
  bit-precise within fp32.

- **`ROADMAP_LLAMA.md`** updated: items 1 (RoPE), 2 (weight-name
  mapping), and now the "actually run it" milestone are crossed
  off. The remaining gaps to a real `llama-7b-Q4_K_M.gguf` are:
    - Q4_K matvec in the backend (currently CPU backend's matvec is
      f32-only; the Q4_K kernel exists in `quant.c` and just needs
      backend hookup) — easy, ~50 lines
    - SentencePiece tokenizer or cheat-mode int32 prompts — easy
      if cheat mode (~30 lines)
    - GQA for ≥13B-class models — medium (~150 lines)
    - Multi-thread Q4_K matvec — performance, not correctness

Total test count: 31/31 passing.

### Added — LLaMA-format GGUF loader (`L9.3`)

slate can now open a llama.cpp-format GGUF file and resolve its
weight tensors + hyperparameters via the standard naming convention.
Real LLaMA-2 / LLaMA-3 / Mistral / Qwen GGUF files share the same
schema so this is the foundation for actually running those models.

- **`include/slate/llama.h`** + **`src/loader/llama.c`**: parses the
  `llama.*` kv metadata namespace (block_count, embedding_length,
  feed_forward_length, attention.head_count, attention.head_count_kv
  with MHA fallback, context_length, vocab_size, rope.freq_base,
  attention.layer_norm_rms_epsilon).  Resolves the standard
  per-block tensor names (`blk.N.attn_q.weight`, etc.) and the
  global tensors (`token_embd.weight`, `output_norm.weight`,
  `output.weight`).  Auto-detects tied output projection (when
  `output.weight` is absent, aliases to `token_embd.weight`).

- **GGUF reader extension** (`src/loader/gguf.c`): KV metadata is now
  retained (not just `general.alignment`) and exposed via three
  accessors: `slate_gguf_get_u32`, `slate_gguf_get_f32`,
  `slate_gguf_get_str`.  This unlocks any future model-format wrapper
  that needs to read GGUF metadata.

- **`tools/make_tiny_llama_gguf.py`**: emits a 100 KB synthetic
  LLaMA-format GGUF (V=64, D=32, n_layers=2, n_heads=2, head_dim=16,
  FFN=64) with all 21 named tensors and 11 kv metadata entries.  CI
  regenerates before running tests.

- **Test** (`tests/test_llama_load.c`): opens the synthetic fixture,
  verifies every config field matches what Python emitted, every
  global + per-block tensor pointer is non-NULL, dtype is f32 (the
  fixture is f32-only), and tied-output detection reports the
  expected state.

- **`ROADMAP_LLAMA.md`**: item 2 (GGUF→slate weight-name mapping)
  crossed off. Items 3 (GQA), 4 (tokenizer), 5 (tied output —
  partially done, the loader detects it; the inference path needs
  to honour the flag), 6 (multi-thread Q4_K), 7 (Flash Attention)
  remain.

Total test count: 30/30 passing.

### Added — Rotary Position Embedding (RoPE) (`L9.2`)

The biggest blocker on the `ROADMAP_LLAMA.md` list — RoPE is what LLaMA
uses instead of additive position embeddings, and you can't load any
real LLaMA-2/3/Mistral/Qwen GGUF without it.

- **`slate_op_rope(ctx, x, positions, theta_base)`**
  (`include/slate/ops.h`, `src/ops/rope.c`):
  - Input: `x` of shape `[B, T, n_heads, head_dim]` (head_dim must be even)
  - Input: `positions` of shape `[T]` (i32) — per-token position
  - `theta_base` = 10000.0 for LLaMA
  - Standard GGML "non-interleaved" convention: rotates the first
    and second halves of head_dim, so
    ```
    y[..., i]        = x[..., i] * cos(angle) - x[..., i + hd/2] * sin(angle)
    y[..., i + hd/2] = x[..., i] * sin(angle) + x[..., i + hd/2] * cos(angle)
    ```
    where `angle = position * theta_base^(-2i/hd)` for `i in [0, hd/2)`.
  - Full autograd backward (rotation by `-angle`).
- **Tested against Python reference** (`tools/make_rope_ref.py`,
  `tests/test_rope.c`):
  - Forward matches numpy reference at `L_inf = 2.4e-7` (fp32 precision floor)
  - Inverse round-trip `rope(rope(x, +pos), -pos) ≈ x` at `L_inf = 4.8e-7`
- **`ROADMAP_LLAMA.md`** updated: item 1 (RoPE) crossed off. Items
  2–7 (GGUF weight-name mapping, GQA, tokenizer, tied output,
  multi-thread Q4_K matvec, Flash Attention) remain.

Total test count: 29/29 passing.

### Refactored — inference engine routes through `slate_backend_t` (`L9.1`)

The L9.0 backend abstraction was a vtable nobody called.  This milestone
wires `slate_infer_engine_t` to dispatch every compute primitive
(matvec / linear_batch / rmsnorm_row / silu_mul / add_inplace /
embed_lookup / attention_step) through the backend's vtable, and
allocates session-side scratch (KV cache + activations) via
`backend->alloc` / `backend->release`.

After this milestone, retargeting slate to a real GPU is just:

```
slate_module_t* model = ...;
const slate_backend_t* gpu = slate_backend_cuda();  // user fills this in
slate_infer_engine_t* eng = slate_infer_engine_new_ex(
    model, n_layers, d_model, vocab, ffn_hidden, max_seq, gpu);
// Everything else (sessions, prefill, decode_step, batch_step,
// scheduler, HTTP server) just works.
```

Backward-compat: `slate_infer_engine_new` (the legacy 6-arg constructor)
is preserved — it now delegates to `_ex` with `backend=NULL` →
`slate_backend_default()` → CPU.

**Bit-identical correctness preserved**: `test_infer` still reports
`L_inf = 0.000000` on prefill, decode_step, and batched output vs the
training-side forward.  All 28/28 tests pass.

Touched: `src/infer/engine.c` (~120 line diff), `include/slate/infer.h`
(new `slate_infer_engine_new_ex` declaration + forward decl of
`slate_backend`).  Nothing outside the engine needed to change — the
inference server, scheduler, batched API, and quant codepaths all
inherit the backend dispatch transparently.

### Added — Compute-backend abstraction + CPU backend (`L9.0`)

Lays the dispatch layer that lets the inference engine target GPUs
later without re-architecting the engine.  This milestone delivers
the abstraction + a fully-tested CPU backend; CUDA / Metal backends
are explicit stubs.  Honesty: writing GPU kernels in a sandbox with
no GPU hardware would ship "looks correct" code with bugs that don't
surface until production.  Slate prefers the empty stub to that.

- **`include/slate/backend.h`**: `slate_backend_t` — a vtable of 13
  function pointers covering everything the inference fast-path
  needs:
    - memory: `alloc`, `release`, `copy_h2d`, `copy_d2h`;
    - compute: `matvec`, `linear_batch`, `rmsnorm_row`, `silu_mul`,
      `add_inplace`, `embed_lookup`, `attention_step`;
    - sync: `sync`.
- **`src/backend/backend_cpu.c`**: full CPU implementation wrapping
  the existing AVX2 kernels + packed-panel GEMM.  64-byte aligned
  `alloc`, trivial `copy`, all primitives match slate's existing
  performance characteristics.
- **`src/backend/backend_cuda_stub.c` + `backend_metal_stub.c`**:
  explicit NULL-returning stubs.  Each carries a comment block
  explaining the recommended implementation pathway (cuBLAS sgemm
  for matvec/linear_batch, single-block reduction kernels for
  RMSNorm/softmax, etc.) so a GPU engineer dropping in real kernels
  has the contract spelled out.
- **`docs/GPU_BACKEND.md`**: design notes + per-primitive
  implementation table + the conformance contract (the test suite
  the GPU kernels must pass) + suggested ordering of future work
  (engine refactor → CUDA backend → multi-GPU sharding → Flash
  Attention 2 → fp16 weights).
- **`tests/test_backend.c`**: 8 conformance tests covering every
  compute primitive against scalar reference.  Max measured drift
  on CPU backend: `4.77e-7` (silu_mul, AVX2 sigmoid polynomial);
  most primitives are `0.000000` or `< 1e-7`.

After this milestone, retargeting slate to a real GPU is:
  1. Refactor `src/infer/engine.c` to call through `slate_backend_t`
     instead of its private static helpers (~150 lines diff, no
     algorithmic change);
  2. Write `backend_cuda.cu` (`backend_metal.m`) against the
     test_backend conformance contract;
  3. CMake conditional substitution wires the real backend in
     place of the stub when `-DSLATE_ENABLE_CUDA=ON`.

Total test count: 28/28 passing.

### Added — GGUF Q4_K_M load path + honest LLaMA gap (`L8.7`)

- **GGUF reader recognises Q4_K_M** (`src/loader/gguf.c`):
  - `GGML_T_Q4_K = 12` tensor type
  - `dtype_nbytes(Q4_K, n) = (n/256) * 144`
  - maps to new `SLATE_DTYPE_Q4_K = 18`
  - `slate_dtype_name` returns `"q4_k"`,
    `slate_dtype_is_quantized` returns true
- **Synthetic Q4_K_M GGUF fixture** (`tools/make_q4k_gguf.py`):
  produces a 2.4 KB GGUF with one `weight` tensor of shape `[16, 256]`
  (16 super-blocks of 256 = 4096 weights total) plus a sidecar
  `slate_q4k_expected.f32` file containing the spec-compliant
  reconstruction. CI regenerates both before running tests.
- **End-to-end test** (`tests/test_gguf_q4k.c`):
  - opens the GGUF via mmap;
  - resolves the `weight` tensor via `slate_gguf_get_tensor`;
  - verifies `dtype == SLATE_DTYPE_Q4_K`;
  - runs `slate_dequant_q4_k` and asserts `L_inf = 0` vs the reference
    reconstruction (bit-identical);
  - runs `slate_dot_q4_k_f32` with a random `y` and compares to
    double-precision `dequant·y` — rel_err 1.4e-7;
  - runs `slate_q4_k_matvec` across all 16 rows and compares to
    `dequant ⋅ x` row-by-row — rel L_inf 1.5e-6.
- **`ROADMAP_LLAMA.md`**: written honestly. Lists the seven things
  still missing for slate to load a real `llama-7b-Q4_K_M.gguf`:
  1. RoPE (biggest blocker)
  2. GGUF→slate weight-name mapping
  3. GQA
  4. SentencePiece tokenizer (or cheat mode for benchmarks)
  5. Tied output projection
  6. Multi-thread Q4_K matvec
  7. Flash-Attention tiling
  with effort estimates and the order I'd suggest tackling them.
  Bottom line: 1.5–2 more focused sessions before the first real
  LLaMA-7B token comes out of slate.

Total test count: 27/27 passing.

### Added — Q4_K_M quantized inference (`L8.5`)

GGML's 4.5-bits/weight super-block format — the deployment-standard
quant for LLaMA-7B class models on consumer hardware.  Adds dequant,
fused inner-product, and matvec entry points alongside the existing
Q8_0 family.

- **Format**: 256-element super-block, 144 bytes per block.
  - `f16 d` + `f16 dmin` super-block scales (4 bytes)
  - 12-byte packed `(scale_j, min_j)` pairs for 8 sub-blocks of 32 weights
    (6-bit unsigned each; standard GGML packing)
  - 128 bytes of 4-bit quants (low nibble = first 16 weights, high = next 16)
  - Reconstruction: `x_i = d · scale_j · q_i − dmin · min_j`

- **`slate_dequant_q4_k`**: spec-conformant dequantizer.  Bit-identical
  to the Python reference in `tools/make_q4k_test.py`
  (verified `L_inf = 0.000000` in `test_q4k`).

- **`slate_dot_q4_k_f32`**: fused Q4_K × f32 inner product using the
  closed-form sub-block factorisation
  ```
  <x, y> = sum_j ( d·scale_j · sum_{i in j} q_i·y_i
                 − dmin·min_j · sum_{i in j} y_i )
  ```
  so weights are never materialised as f32.  AVX2 path: load 16
  nibble-bytes → unpack to 32 int8s → convert to f32 → FMA with y.
  Verified within 1.5e-7 relative error vs dequant-then-dot.

- **`slate_q4_k_matvec`**: per-row wrapper over the dot kernel, same
  shape as `slate_q8_0_matvec` so calling code can swap quant formats.

- **Benchmark** (`benchmarks/bench_q4k_vs_q8.c`), 4096×4096 matvec
  (LLaMA-7B attention/FFN projection shape):

  | format | memory      | latency      | GFLOP/s | vs f32 |
  |--------|------------:|-------------:|--------:|-------:|
  | f32    | **64.0 MB** | 18.40 ms     | 1.82    | 1.0×   |
  | Q8_0   | 17.0 MB     | 3.55 ms      | 9.44    | 5.18×  |
  | Q4_K_M | **9.0 MB**  | **3.17 ms**  | **10.59** | **5.81×** |

  Memory **7× smaller than f32, 1.9× smaller than Q8_0** — matching
  the spec ratio (8.25 vs 4.5 bits/weight).  After the AVX2 kernel
  optimisation (see "Q4_K AVX2 kernel" below) Q4_K_M is now **1.12×
  faster than Q8_0** on top of being half the memory: the smaller
  weights save more L2 bandwidth than the extra sub-block bookkeeping
  costs.  At LLaMA-7B model size (~6.5B params), f32 weighs ~26 GB;
  in Q4_K_M it fits in **~3.8 GB** — runnable on a 4 GB Raspberry Pi 5.

### Performance — Q4_K AVX2 kernel

- The first-pass Q4_K dot kernel issued **16 `hsum256` calls per
  super-block** (one for `qy` and one for `sy` in each of the 8
  sub-blocks).  Horizontal SIMD reductions are expensive — they stall
  on cross-lane shuffles.
- Rewrote the inner loop to **defer the hsum to the super-block
  boundary** (2 hsums per super-block instead of 16):
  - Per sub-block: collapse the 4 partial `__m256` accumulators down
    to a single `__m256` `qy_p` (and `sy_p`) using SIMD lane-wise
    adds — no horizontal reduction.
  - Broadcast the scalar `scale_j` / `min_j` into all 8 lanes and FMA
    into super-block-wide `__m256` accumulators (`d_acc`, `m_acc`).
  - After all 8 sub-blocks, hsum once per accumulator.
- Result: **Q4_K_M dot throughput 6.42 → 10.59 GFLOP/s (1.65× faster)**.
  Test_q4k still passes with `L_inf = 0.000000` vs the Python
  reference and `rel_err = 7.97e-8` vs dequant-then-dot — i.e.
  correctness preserved.

- **Test** (`tests/test_q4k.c`): three-part conformance check:
    1. Dequant matches Python reference bit-identically;
    2. Dequant within 0.05 mean-abs error of the original f32 weights;
    3. Direct dot matches dequant-then-dot within 1e-5 relative error.

- **Fixture generator** (`tools/make_q4k_test.py`): produces the test
  block + reference reconstruction from a known seed, so the test
  fixture is reproducible and the CI can regenerate it.

Total test count: 26/26 passing.

### Added — Server-side batching scheduler (`L8.4`)

Wires the L8.3 batched engine into the production HTTP server so end-to-end
throughput improves under concurrent load, not just on synthetic benchmarks.

- **`slate_scheduler_t`** (`include/slate/scheduler.h`,
  `src/server/scheduler.c`): one dedicated decoder thread + a shared
  pending-request queue. Workers call
  `slate_scheduler_decode(scheduler, sess, token, out_logits)`,
  which is synchronous from the worker's POV: it queues the request,
  blocks on a per-request cond var, and returns when the decoder has
  drained up to `max_batch` requests in one `slate_infer_batch_step`
  call and filled `out_logits`. Workers can leave the loop between
  tokens (max_tokens hit, EOS, etc.) without breaking the batch.
  Atomic stats track total batches and average batch size for
  observability.

- **HTTP server integration** (`src/server/http.c`): when
  `slate_server_config_t::scheduler_max_batch > 0` (default 16), the
  decode loop in `handle_connection` routes through
  `slate_scheduler_decode` instead of calling `slate_infer_decode_step`
  directly. Per-session prefill stays on the worker (it's cheap, and
  each session is independent). Streaming and non-streaming responses
  both benefit. Same auth + rate-limit + metrics path.

- **Benchmark** (`benchmarks/bench_server_load.c`): real HTTP load
  test — spawns one slate server, then 8 client threads each making
  4 sequential /v1/completions requests for 16 generated tokens each.
  Measured (model V=4096 D=256 L=4 FFN=1024, AVX2 single-thread sandbox):

  | scheduler | total tokens | wall time | tok/s    |
  |-----------|-------------:|----------:|---------:|
  | off       | 512          | 1.289 s   | **397**  |
  | on (B=16) | 512          | 0.722 s   | **710**  |

  **1.79× end-to-end** on real concurrent load. The headroom toward
  the synthetic-benchmark 9.7× comes from:
    1. prefill is still per-request (not yet batched);
    2. 16-token generations are short — fixed per-request HTTP
       overhead is a larger fraction than at production-grade 200+
       token generations;
    3. only 8 concurrent clients in this run — the scheduler's
       max_batch is 16, so it's running at half capacity for this
       configuration.

### Added — Continuous batching (`L8.3`)

Batched inference path for multiple concurrent sessions. The biggest
single throughput win in the production stack — for small models the
per-call overhead of the packed-panel GEMM dominates at M=1, and
stacking sessions into a single M=B GEMM amortises that overhead
nearly perfectly.

- **`slate_infer_batch_t` + `slate_infer_batch_step`**
  (`include/slate/infer.h`, `src/infer/engine.c`): pre-allocated
  batch-wide scratch for B concurrent sessions. One call advances all
  B by one token. The linear projections per layer (Wq, Wk, Wv, Wo,
  Wg, Wu, Wd, lm_head) all run as M=B GEMMs; only the causal
  attention stays per-session (each request has its own cache
  length, which is exactly the *point* of continuous batching).
  Sessions in a batch can be at *different* current positions — the
  engine looks up `pos_emb[sess->position]` and runs full
  cache-aware attention per session, so the scheduler can mix
  long-context with fresh prompts in the same batch.

- **Correctness contract**: `tests/test_infer.c` now includes a
  batched-vs-sequential equivalence test. Same prompts, same
  next-tokens; batched output's L_inf = 0.000000 vs sequential, i.e.
  bit-identical. KV caches remain fully independent across sessions
  — they are never aliased.

- **Throughput measured** (`benchmarks/bench_batch_throughput.c`,
  model: V=4096 D=256 L=4 FFN=1024, AVX2 single-thread sandbox):

  | B  | sequential tok/s | batched tok/s | speedup |
  |---:|-----------------:|--------------:|--------:|
  |  1 |              320 |           318 |   1.00× |
  |  2 |              325 |           623 |   1.95× |
  |  4 |              328 |          1188 |   3.72× |
  |  8 |              317 |          2215 |   6.93× |
  | 16 |              319 |          3100 |   9.71× |

  Sequential throughput stays flat at ~320 tok/s regardless of B
  (confirming the M=1 GEMM overhead bottleneck); batched throughput
  scales nearly linearly to ~3100 tok/s at B=16. That's the line
  that takes slate from "demo" to "actually serve concurrent users
  on one CPU".

### Added — Production CPU inference server, round 2 (`L8.2`)

Production-readiness pass on the inference server added in L8: streaming,
multi-tenant auth, rate limiting, graceful shutdown, and CI.

- **SSE token streaming on `/v1/completions`**: when the request body
  carries `"stream": true`, the server switches from buffered JSON to
  `Content-Type: text/event-stream` and emits one `data: {"token": N}`
  event per generated token, followed by a `data: [DONE]` terminator.
  Same compute path as the non-streaming case (one engine
  `decode_step` per token) — the difference is whether the response
  is collected before the headers are written. Compatible with the
  OpenAI streaming protocol shape. Latency to first token is recorded
  as a separate `slate_time_to_first_token_ms` histogram so an
  operator can SLO on prompt-prefill cost independently of total
  generation time.

- **Multi-tenant auth + per-key rate limit** (`include/slate/apikey.h`,
  `src/server/apikey.c`): `slate_apikey_set_t` holds N keys, each with
  its own token-bucket (rps refill + burst capacity). Authentication
  matches the `Authorization: Bearer ...` header against the set; on
  match the bucket is decremented and the request proceeds. Empty
  bucket → HTTP 429 with `slate_rate_limited_total` counter
  incremented; not found → 401 as before. One mutex per key so
  bursts on key A don't serialise against key B traffic. Schema +
  loader for JSON config files so secrets aren't compiled in:
  ```json
  [
    {"key": "...", "label": "free-tier",  "rps": 1.0, "burst": 5},
    {"key": "...", "label": "paid-tier",  "rps": 0,   "burst": 0}
  ]
  ```
  The single-key `api_key` field on the server config still works for
  back-compat; if present (and `apikey_set` is NULL) it's wrapped in a
  synthetic single-entry set with no rate limit.

- **Graceful SIGINT / SIGTERM shutdown**:
  `slate_server_install_signal_handler()` wires both signals to
  `slate_server_stop()`. On stop: the accept loop exits, `/health`
  starts answering 503 (so an upstream LB can pull the node out of
  rotation), workers drain pending requests, and
  `slate_server_free()` waits up to `shutdown_timeout_sec` for the
  in-flight count to reach zero before joining workers. Hard timeout
  forces termination and logs `shutdown_timeout` with the remaining
  request count.

- **More metrics**: `slate_rate_limited_total`,
  `slate_stream_requests_total`, `slate_active_requests` (gauge — used
  for the graceful drain), `slate_time_to_first_token_ms` histogram.
  Request log records now include `key`, `stream`, and `ttft_ms`.

- **Inference engine introspection**: added
  `slate_infer_engine_vocab(eng)` and `slate_infer_engine_max_seq(eng)`
  getters so callers can size logits buffers and clamp sampled token
  ids to the actual vocab without having to thread those constants
  through manually. Caught a real bug in the L8 server during
  testing — without the getter, sampling against an oversized buffer
  was returning out-of-vocab ids and `decode_step` was rejecting
  them after the first token.

- **GitHub Actions CI** (`.github/workflows/ci.yml`): builds slate on
  Ubuntu latest with both Release and Debug; regenerates the GGUF
  fixtures (`tools/make_test_gguf.py`, `tools/make_q8_gguf.py`);
  runs the full ctest suite; runs the inference server example as a
  real end-to-end smoke test. Separate ASAN/UBSAN job catches memory
  / undefined-behaviour regressions on every push.

- **Example updates**: `examples/11_inference_server` now demonstrates
  all the new features end-to-end — registers two keys (one
  rate-limited, one unlimited), hits both, triggers a 429 by exceeding
  the rate-limited key's burst, hits the streaming endpoint and
  parses the SSE events, hits `/health` again after a stop to see the
  draining response.

### Added — Production CPU inference server (`L8`)

End-to-end serving stack: a model trained with slate goes through the
new `slate_infer_engine_t` (KV-cached, no autograd graph), behind a
hand-rolled HTTP/1.1 server with Prometheus metrics + structured JSON
logging + bearer-token auth.

- **`include/slate/infer.h` + `src/infer/engine.c`**:
  inference-only forward path with a per-session KV cache. Single-step
  decode is `O(L·D)` instead of the training forward's `O(L²·D)`, so
  generation latency stays roughly linear past a long prompt. Same
  packed-panel AVX2 GEMM as training; same RMSNorm / SwiGLU SIMD
  helpers. Bypasses the autograd graph entirely — no node arenas, no
  gradient buffers, no scratch tensors. Engine is read-only after
  construction (shareable across threads); sessions are per-request.
  Memory cost per session: `2 · n_layers · max_seq · D · 4 B`.
  Bit-identical to the training-side forward — tested in `test_infer`
  with `L_inf = 0` on both prefill and decode_step paths.

- **`include/slate/server.h` + `src/server/http.c`**: minimal HTTP/1.1
  server on raw BSD sockets (no libcurl / libmicrohttpd). Endpoints:
  `POST /v1/completions` (JSON request: `prompt`, `max_tokens`,
  `temperature`, `top_p`, `top_k`), `GET /health`, `GET /metrics`
  (Prometheus exposition). One thread per connection from a fixed-size
  worker pool; each worker holds its own `slate_infer_session_t` so KV
  caches don't cross requests. `Authorization: Bearer <key>` when an
  `api_key` is configured; bound to `INADDR_ANY` so a TLS-terminating
  reverse proxy (nginx / Envoy / Caddy) can sit in front.

- **`include/slate/metrics.h` + `src/server/metrics.c`**: Prometheus
  counter / gauge / histogram. Atomic adds for thread safety, single
  registry lock for the construction path. Exposition emits a stable
  Prometheus text format consumable by the standard scrape pipeline.
  Built-in server metrics: `slate_requests_total`,
  `slate_tokens_in_total`, `slate_tokens_out_total`,
  `slate_auth_failures_total`, `slate_errors_total`,
  `slate_active_connections` (gauge), `slate_request_latency_ms`
  (histogram, 14 buckets `1ms..8s`), `slate_tokens_out_hist`.

- **`include/slate/jlog.h` + `src/server/jlog.c`**: structured (JSON)
  one-line-per-record logger on stderr. Pluggable level filter,
  pluggable output `FILE*`. Standard fields: `ts` (RFC3339 UTC),
  `level`, `event`, `fields{}`. Designed to be ingested by
  Loki / OpenSearch / Splunk without further parsing.

- **`examples/11_inference_server/`**: end-to-end demo. Builds a tiny
  random-weight transformer, wraps it in the engine, runs the server
  on a background thread, then a built-in self-client hits `/health`,
  `/v1/completions` both with and without auth, `/metrics`, and an
  unknown URL — verifying status codes (200 / 401 / 200 / 404), JSON
  payload shape, and the metrics output (`slate_requests_total`,
  `slate_auth_failures_total` correctly incremented). Exits 0 on full
  pass.

- **`tests/test_infer.c`**: KV-cache correctness contract. Compares
  the engine's prefill output against `slate_module_forward` on the
  same prompt + last-position slice, then compares a single
  `decode_step` after that prefill against a full re-forward on the
  extended prompt. Both Linf differences must be < 1e-3 (actually
  observed: 0.000000 on both, i.e. bit-identical).

Open production gaps not yet addressed (see ROADMAP): continuous
batching across concurrent requests, paged-attention style cache
eviction for long conversations, Q4_K_M / Q5_K quantized inference,
TLS termination in-process, structured rate-limiting per-key, model
hot-reload, GPU back-end.

### Performance — packed-panel GEMM (kernel-layer rewrite)

The forward and backward matmul were rewritten to GotoBLAS-style three-level
blocking (`MC × KC × NC`) with a 8×8 AVX2 register microkernel, software
prefetch into L1, and per-thread persistent packing scratch (`__thread`):

- Tile parameters: `MR=8`, `NR=16`, `MC=64`, `KC=128`, `NC=512` — sized so the
  A panel (32 KiB) lives next to L1 and the B panel (256 KiB) fits in L2 on
  a typical consumer CPU.
- Microkernel (final 8×16 form): **16 YMM accumulators** (2 per row, low+high
  halves of NR=16) — saturates the full AVX2 register file. Per K step:
  2 B-loads + 8 A-broadcasts + 16 FMAs. Full-tile fast-path with edge
  spill-and-merge for non-multiples of 8/16.
- An intermediate 8×8 microkernel (8 accumulators) reached ~42 GFLOP/s on
  1024³ single-thread; upgrading to 8×16 lifted that to ~66 GFLOP/s (~1.57×).
- Backward: `d_a = d_out @ b^T` and `d_b = a^T @ d_out` are turned into
  standard GEMMs via an out-of-place transpose, so they go through the same
  packed kernel and stay threaded.
- Per-thread `__thread` packing buffers eliminate the malloc-per-call
  overhead that originally hurt small-matrix performance under threading.

Measured on the 2-core x86_64 sandbox at 1024×1024×1024:

|                  | baseline    | 8×8 µkernel  | 8×16 µkernel | total speedup |
|------------------|-------------|--------------|--------------|---------------|
| 1-thread AVX2    | 14 GFLOP/s  | 42 GFLOP/s   | **66 GFLOP/s** | **~4.7×**   |
| 2-thread AVX2    | 28 GFLOP/s  | 81 GFLOP/s   | **110 GFLOP/s**| **~3.9×**   |

At 1024³ single-thread, the kernel reaches **roughly 70 % of the AVX2
theoretical peak** for the sandbox clock — comparable to mature
implementations like ggml's. Numbers will vary with CPU and turbo
behavior on real hardware.

**bmm (batched matmul) shares the same kernel.** The forward / backward
were refactored to call into `src/ops/gemm_internal.h::slate_gemm_packed_accumulate`
on a per-batch basis (parallelism moved from inside-the-matmul to
across-the-batch). For transformer attention shapes this means:

|  Shape (BxHxSxD)         | OP        | 1-thread (8×16) |
|--------------------------|-----------|-----------------|
| 1·12·256·64 ([12,256,64]×[12,64,256])   | Q @ Kᵀ    | ~62 GFLOP/s    |
| 1·12·256·64 ([12,256,256]×[12,256,64])  | attn @ V  | ~42 GFLOP/s    |
| 1·8·512·64                              | Q @ Kᵀ    | ~64 GFLOP/s    |
| 8·4·64·32  (char-LM)                    | Q @ Kᵀ    | ~40 GFLOP/s    |

Compared to the previous scalar-AXPY `mm2d` inside `bmm.c` (which was
essentially the pre-optimisation matmul ~14 GFLOP/s), per-batch attention
GEMMs are now **~3–4× faster on forward**. The backward, which was previously
a fully scalar `O(L·M·K·N)` triple loop, now runs through the same packed
kernel via materialised transposes — about **5–8× faster** depending on shape.

While reworking matmul, a latent gradient-correctness bug was caught: the
multi-threaded `d_b = a^T @ d_out` path had its `M` shadowed by the per-task
band size, so `aT`'s row stride was wrong whenever `nt > 1`. Single-thread
runs (which gradcheck used) were correct; the issue only triggered with
≥2 threads and was fixed during the bmm refactor.

All 18 tests (including `test_gradcheck` — finite-difference validation of
all eight gradient-bearing ops — `test_transformer_block` — end-to-end
TransformerBlock forward+backward — and `test_mha` — multi-head attention,
which exercises the bmm rewrite end-to-end) still pass. The Final-assembly
capstone still produces `adapter byte-identical after disk round-trip: yes`
and `GGUF base file unchanged (byte-level): yes`.

Benchmarks: `benchmarks/bench_matmul.c` (2-D GEMM), `benchmarks/bench_bmm.c`
(batched attention shapes).

### Performance — Q8_0 × f32 direct dot (inference path)

`slate_dot_q8_0_f32` and `slate_q8_0_matvec` (in `include/slate/quant.h` /
`src/util/quant.c`) fuse the dequant + dot product into a single AVX2 pass
over the packed 34-byte Q8_0 blocks. Previously, a Q8_0 × f32 matvec had
to dequantise the entire weight matrix to a temporary f32 buffer first,
then run a standard dot — paying for one full f32 write and one full f32
read of a matrix that may be hundreds of MiB.

Per-block AVX2 inner loop: load 32 int8 weights → split into two 16-int8
halves → sign-extend to int16 then to int32 → convert to f32 → multiply
by the f16 scale once → four FMA accumulations against 32 floats of `x`.
Final reduction is a horizontal sum of one __m256 accumulator.

Numbers (single thread, AVX2, sandbox):

| Shape (M × K)      | dequant + f32 dot  | fused Q8 dot       | speedup |
|--------------------|--------------------|--------------------|---------|
| 64 × 256           | 0.04 ms / 0.9 GF/s | 0.01 ms / 5.7 GF/s | 6.2×    |
| 256 × 1024         | 0.43 ms / 1.2 GF/s | 0.05 ms / 10.4 GF/s| 8.5×    |
| 1024 × 1024        | 1.32 ms / 1.6 GF/s | 0.21 ms / 10.0 GF/s| 6.3×    |
| 4096 × 4096 (7B-class) | 30.6 ms        | 3.6 ms / 9.4 GF/s  | **8.6×**|

The absolute throughput is bandwidth-bound (Q8_0 still reads the entire
weight matrix once); the win comes from eliminating a parallel f32
write-then-read of the same matrix. For LLaMA-7B style 4096×4096 attention
weights this is the dominant inference cost.

Verified bit-equivalent to the dequant+dot reference in `test_q8_dot`
(relative error ≤ 1e-6, only floating-point summation order differs).
Benchmark: `benchmarks/bench_q8_dot.c`.

### Performance — softmax and RMSNorm AVX2 paths

Both `slate_op_softmax` and `slate_op_rms_norm` now have AVX2-vectorised
inner loops. They were previously pure scalar with a `expf()` and `sqrtf()`
call per element. For the SIMD path we also use:

- A **vectorised exp polynomial** for softmax (`exp256_ps`): range reduce
  `x = n·ln(2) + r` then a degree-5 polynomial for `exp(r)` on `|r| ≤ ln(2)/2`,
  scaled by `2^n` via integer add into the IEEE-754 exponent field. Accuracy
  ≈ 1 ulp in the reduced range; max relative error ≈ 3e-7.
- **Double-precision scalar tails** for the reduction inner products so that
  `d_y[i] - s` in the softmax backward does not lose precision when
  individual gradients land near the 1e-7 noise floor.

To keep the gradient-check tests bit-stable, both ops keep their **original
scalar code path verbatim** for last-dim sizes below 16 (the `SIMD_MIN_C`
threshold). Transformer shapes — attention scores, LM head, hidden state —
are all far above 16 and take the AVX2 path.

Measured (single thread, AVX2, sandbox):

| op       | shape                              | per-row time | throughput     |
|----------|------------------------------------|--------------|----------------|
| softmax  | N=8192, C=128 (attention S=128)    | 0.12 us      | 1.09 Gelems/s  |
| softmax  | N=3072, C=256 (attention S=256)    | 0.22 us      | 1.15 Gelems/s  |
| softmax  | N=4096, C=512 (attention S=512)    | 0.48 us      | 1.07 Gelems/s  |
| softmax  | N=10,   C=32000 (LM head vocab=32k)| 25.6 us      | 1.25 Gelems/s  |
| RMSNorm  | N=512,  C=128 (hidden=128)         | 0.046 us     | 2.78 Gelems/s  |
| RMSNorm  | N=512,  C=768 (GPT-2 sm)           | 0.245 us     | 3.13 Gelems/s  |
| RMSNorm  | N=512,  C=4096 (LLaMA-7B)          | 1.77 us      | 2.32 Gelems/s  |

RMSNorm at >2 Gelems/s is essentially memory-bandwidth-bound. The softmax
throughput is exp-bound; the polynomial approximation gets roughly 3× over
glibc `expf`.

Benchmark: `benchmarks/bench_softmax_rmsnorm.c`.

While reworking softmax a numerically sensitive case was caught: the
test_gradcheck softmax shape (2×3, max gradient component ~1e-7) requires
double-precision accumulation in the sum reduction. The earlier SIMD draft
which used float accumulation lost cancellation precision and tripped the
1e-2 relative-error threshold. The final code carries the reduction in
double in both the SIMD and scalar paths.

### Performance — SiLU, add, mul, add_bias, scale (remaining element-wise)

The remaining hot element-wise / activation ops were vectorised too:

- **SiLU (`y = x * sigmoid(x)`)**: forward uses the vectorised
  `slate_sigmoid256_ps` (built on `slate_exp256_ps`); backward is straight
  AVX2 FMA. Every transformer FFN block calls SiLU twice per token.
- **`add` / `mul` (elementwise)**: forward and backward are simple
  vectorised loops; `mul` backward uses `_mm256_fmadd_ps` to merge the
  `dy * other_operand` with the existing grad accumulator.
- **`add_bias` (the `Linear(bias=True)` post-step)**: 8-wide broadcast-add
  on the bias vector; backward sums upstream gradient over N rows into
  `d_b` 8-wide.
- **`scale` (the `1/sqrt(d)` factor in attention)**: forward is a vector
  multiply by a broadcasted scalar; backward is an FMA into the grad
  accumulator.

A new private header `src/ops/simd_helpers.h` collects the shared
intrinsics — `slate_hsum256`, `slate_hmax256`, `slate_exp256_ps`,
`slate_sigmoid256_ps` — that softmax, RMSNorm, and SiLU all consume,
so there is exactly one copy of the polynomial exp.

All 19 tests still pass (`test_gradcheck` covers every op with finite
differences, `test_transformer_block` exercises SiLU + add + add_bias +
scale + RMSNorm end-to-end), and the Final-assembly capstone is still
byte-identical.

### Added — On-Policy Distillation (OPD)

- **`slate_op_kd_loss_topk`** (`include/slate/kd.h`, `src/objective/kd.c`):
  KL teacher→student loss where the teacher distribution is given only on
  its top-K vocab indices per position, with everything else treated as
  zero support. Gradient comes out to `T·(Q_v − P̃_v) / (B·T)` per slot
  with `P̃` the sparse (top-K only) teacher distribution — identical
  shape to the dense `slate_op_kd_loss` backward, just with a sparse
  subtraction. Saves ~2× memory over the dense variant for K << V and
  matches the shape of remote teacher payloads (HTTP, mmap'd cache).
- **`slate_topk_extract`** (`include/slate/opd.h`, `src/objective/opd.c`):
  one-shot O(V·K) per-position top-K extraction over a `[B, T, V]`
  logits buffer. Used to build the inputs to `kd_loss_topk` from a
  teacher forward pass.
- **OPD training recipe**: documented in `include/slate/opd.h` and
  exercised end-to-end in `tests/test_opd.c` and `examples/10_opd/`.
  Pipeline per step:
  1. Run the student in eval-mode on the current sequence; sample the
     next token via `slate_sample_token`; append; repeat.
  2. Re-run the student on the full rollout with `training=true`; the
     output logits have `requires_grad`.
  3. Run the (frozen) teacher on the same rollout — since teacher
     parameters are not in the optimizer's param set, no update flows
     back to them.
  4. `slate_topk_extract` on teacher logits → `[B, T, K]` indices +
     logits tensors.
  5. `slate_op_kd_loss_topk` → backward → optimizer step.
  Same code structure works with the embedding-style "Markov LM" used
  in the demo and with `slate_module_causal_lm` transformers (just
  swap the forward call). Pairs naturally with Muon for matrix weights
  + AdamW (or Muon's SGD-momentum fallback) for biases/embeddings.
- **Test (`test_opd`)**: synthetic peaky teacher in `[V, V]` form;
  student starts uniform; after 250 OPD steps, fixed-eval argmax
  agreement reaches V/V (8/8), KD loss drops > 30×.
- **Example (`examples/10_opd/`)**: same Markov-style teacher with
  V=16, K=4. Trains in ~3 s and shows eval-KL dropping 3× while
  argmax-agree goes from 1/16 to 16/16. The example prints the
  rollout-loss vs the fixed-eval KL side-by-side and explains in a
  comment why on-policy rollout-loss is *not* monotone (the basis
  itself moves between steps) and the fixed-eval metric is what to
  watch for convergence.

### Added — Muon optimizer (Newton-Schulz orthogonalised momentum)

- **Muon** (`include/slate/optim.h`, `src/optim/muon.c`): the
  MomentUm Orthogonalized by Newton-Schulz optimizer from Jordan et al.
  (2024). On 2D parameter matrices, after the usual momentum + Nesterov
  shift, the update direction is normalised by its Frobenius norm and
  pushed through a 5-step quintic Newton-Schulz iteration
  (`X ← aX + (bA + cA²)X` with `A = XXᵀ`, `(a,b,c) = (3.4445, -4.7750,
  2.0315)`), then scaled by `max(1, √(rows/cols))`. The result is an
  update whose singular values are all ≈ 1, i.e. as close to a true
  orthogonal direction as a quintic can cheaply produce.
- For non-2D parameters (biases, embedding tables, LayerNorm gains,
  conv tensors, etc.) Muon falls back to SGD-with-momentum, matching
  the reference Python implementation's "Muon for matrices, AdamW for
  everything else" recipe — but folded into a single optimizer for
  convenience.
- The Newton-Schulz iteration's matmuls dispatch to the same
  `slate_gemm_packed_accumulate` kernel that powers `slate_op_matmul`,
  so NS cost on a 4096×4096 weight is ~5 × 137 GFLOP ≈ 11 ms per step
  at ~60 GFLOP/s — negligible against a transformer's forward+backward.
- A single shared NS scratch is allocated once at construction, sized
  to the largest 2D parameter — so per-parameter optimizer state is
  just one momentum buffer (1× the parameter footprint, same as SGD,
  half of AdamW).
- AVX2 inner loops for the per-element parts: momentum update,
  Nesterov shift, Frobenius normalisation, the `bA + cA²` linear
  combination, and the final weight update.
- Tested in `tests/test_muon.c` across three shapes that exercise the
  branches: tall `[4, 1]` (rows > cols, transpose-and-restore path);
  wide `[2, 5]` (rows < cols, direct path); `[3, 2]` + 1D bias (mix of
  NS path and SGD-momentum fallback). Loss drops > 5× on all three.

### Summary — what is now SIMD vs still scalar

After the kernel-optimisation pass, the SIMD-vectorised ops are:
`matmul`, `bmm`, `softmax`, `rms_norm`, `silu`, `add`, `mul`, `add_bias`,
`scale`, `dot_q8_0_f32` / `q8_0_matvec` (Q8_0 direct inner product).

Still scalar (lower priority — either rarely called, memory-bound, or
intrinsically hard to vectorise): `activation` (sigmoid/relu/tanh),
`embedding` (random-access gather — memory-bound), `causal_mask` (mostly
predicate writes), `cross_entropy` (one call per step), `loss` (likewise),
`linear3d` (delegates to matmul internally), `matmul_bf16` (bf16 path,
already exists for memory savings not for throughput), `permute12`,
`transpose`. These together account for well under 10% of transformer
forward+backward wall time in our measurements, so further SIMD work
would have a diminishing return.

### Added — Final (engineering completeness)

- **AdapterManager** (`include/slate/adapter_mgr.h`, `src/adapter/adapter_mgr.c`):
  on-disk LoRA adapter lifecycle — atomic write-tmp-then-rename install, archive
  on swap with timestamped name, list/load/promote/rollback. Same-second
  collisions resolve with a `.dup` suffix. Test: `test_adapter_cache`.
- **TeacherCache** (`include/slate/teacher_cache.h`, `src/teacher/teacher_cache.c`):
  append-only binary file of top-k teacher logits keyed by `(token_seq_hash, k)`.
  Lets off-policy KD reuse expensive teacher calls across epochs. Round-trip
  verified bit-identical.
- **NEON kernel path** (`src/runtime/simd_neon.c`): vector add / multiply / dot
  under `#ifdef __ARM_NEON`. Compiles cleanly on aarch64; AVX2 path on x86_64
  is unchanged. Not benchmarked in-sandbox (x86_64 only).
- **Process reward** (`include/slate/process_reward.h`,
  `src/objective/process_reward.c`): step-wise PRM scoring with SUM / MEAN / MIN
  aggregation across CoT steps. Plugs into the GRPO loss as an alternative to
  the outcome-only `RewardFunction`.
- **HTTP teacher stub** (`include/slate/http_teacher.h`,
  `src/teacher/http_teacher.c`): `slate_http_teacher_t` with a user-supplied
  transport callback so the framework stays libcurl-free. Smoke test
  (`test_prm_http`) verifies the request/response plumbing with an in-memory
  fake transport.
- **examples/09_final_assembly**: end-to-end capstone. Loads a synthetic
  4-tensor LLaMA-shape GGUF (Q8_0), mmaps it, dequantizes one block at a time,
  wraps the projection with a `QuantizedLoRA` adapter (2048 trainable params
  out of 16384 base — 1/8 working-set compression), runs a forward + backward
  with gradient checkpointing, saves the adapter via `AdapterManager`, and
  asserts both `adapter byte-identical after disk round-trip: yes` and
  `GGUF base file unchanged (byte-level): yes`.

### Added — M7 (GRPO with DAPO and Dr.GRPO for code)

- **GRPO family of losses** (`include/slate/grpo.h`, `src/objective/grpo.c`):
  `slate_grpo_loss_*` with config flags for token-level vs. sequence-level
  reduction (DAPO), std-normalization on/off (vanilla vs. Dr. GRPO), and
  clip-higher (ε_high > ε_low). Verified by a toy advantage signal:
  vanilla GRPO and Dr.GRPO+DAPO both move policy probability toward the
  positive-advantage trajectory.
- **Reward functions** (`include/slate/reward.h`): pluggable
  `RewardFunction` interface with `TestPassReward`, `CompileReward`,
  `LinterReward`, `CompositeCodeReward`. Toy verified.
- **Subprocess sandbox** (`include/slate/code_executor.h`,
  `src/executor/subprocess.c`): `WasmtimePyodideExecutor` placeholder and a
  working `SubprocessSandboxExecutor` that fork/execs with RLIMIT_AS,
  RLIMIT_CPU, RLIMIT_NOFILE limits and a SIGKILL timeout watchdog. Captures
  stdout/stderr/exit-code/wall-time. Anti-escape: no inherited fds, child
  process group, working directory restricted to a per-run tmp dir.
- **examples/08_code_rl**: toy GRPO loop where the model emits short code
  strings, the sandbox runs them, and `CompositeCodeReward` returns a
  reward; policy probability of the passing program rises across steps.
- New tests: `test_grpo`, `test_sandbox`.

### Added — M6 (Knowledge distillation and RLHF)

- **TrainingObjective interface** (`include/slate/objective.h`): vtable with
  `forward(student_logits, batch) -> (loss, grad_wrt_logits)` so SFT, DPO,
  KTO, KD, and GRPO all plug into the same training loop.
- **DPO** (`include/slate/dpo.h`, `src/objective/dpo.c`):
  `-log σ(β·(logp_chosen - logp_rejected - logp_chosen_ref + logp_rejected_ref))`.
  Toy verified: 200 steps drives P(chosen)=0.5→0.997. Note: prior implementation
  had the gradient sign inverted on chosen/rejected; current code is
  `dc[b] += g; dr[b] += -g;`.
- **KTO** (`include/slate/kto.h`, `src/objective/kto.c`):
  σ(-r) for good examples, σ(r) for bad, with `r = β·(logp - logp_ref)`. Works
  on single examples (no pair required); useful when only a "good / bad" label
  is available.
- **KD** (`include/slate/kd.h`, `src/objective/kd.c`):
  `KL(softmax(teacher/T) || softmax(student/T)) · T²`. Supports off-policy
  (teacher logits from `TeacherCache`) and on-policy (live `Teacher`).
- **Gradient checkpointing** (`include/slate/checkpoint.h`,
  `src/objective/checkpoint.c`): sub-context rematerialization via
  `save_for_backward` + `selective_keep_mask`. Trades roughly 30% extra compute
  for ~50% activation memory in the streaming case.
- New tests: `test_dpo`, `test_kto_kd`, `test_checkpoint`.

### Added — M5 (LoRA + streaming + GGUF + dequant + bf16)

- **LoRA adapter** (`include/slate/adapter.h`, `src/module/lora_adapter.c`):
  `y = base + (α/r) · (x @ A) @ B` with A normal-init and B zero-init.
  Freezes the base parameter; only A and B receive gradients. Verified on
  `examples/05_lora` by training only A/B and asserting base weights are
  byte-identical pre- and post-training.
- **Sub-module streaming** (`include/slate/stream.h`,
  `include/slate/streaming_module.h`, `src/runtime/stream_io.c`,
  `src/module/streaming_linear.c`): mmap-backed `PROT_READ` view of a
  per-layer weight blob, with `StreamingLinear` faulting one sub-block in at a
  time. Peak RAM verified flat in `examples/06_streaming` regardless of the
  on-disk model size (sandbox limit: a 64 MiB stream runs under 8 MiB working
  set).
- **GGUF v3 reader** (`include/slate/gguf.h`, `src/loader/gguf.c`):
  parses headers, KV metadata, and tensor records; returns offsets without
  reading bytes. Verified on synthetic GGUFs produced by
  `tools/make_test_gguf.py`, `tools/make_q8_gguf.py`,
  `tools/make_mini_llama_gguf.py`. Includes `test_gguf`.
- **f16/bf16 conversion** (`include/slate/precision.h`,
  `src/util/precision.c`): IEEE-754 half-to-float and bfloat16 round-trip
  helpers. bf16 path keeps fp32 master weights and casts only for the matmul
  forward (`src/ops/matmul_bf16.c`).
- **Q8_0 and Q4_0 dequantization** (`include/slate/quant.h`,
  `src/util/quant.c`): 34-byte and 18-byte GGML block layouts decoded into
  fp32. End-to-end demo: GGUF → mmap → dequant → matmul against a fp32 input
  matches the f32 reference within 1e-2 relative error.
- **QuantizedLoRA module** (`src/module/quantized_lora.c`): base weight stays
  quantized on disk; only the (A, B) low-rank matrices are dequantized into
  RAM and updated by the optimizer. Used by `examples/07_quantized_lora` and
  by the M5 capstone, which proves the GGUF file is byte-unchanged after a
  full training run.
- **RuntimeMode state machine** (`include/slate/mode.h`,
  `include/slate/mode_state.h`, `src/runtime/mode_state.c`):
  `INFERENCE` / `TRAINING` / `TEACHER_SCORING` with explicit transition rules
  and a session-budget hook so the day/night controller can pause when the
  laptop battery is below threshold or under thermal throttle.
- New tests: `test_gguf`, `test_quant`, `test_gguf_q8_matmul`,
  `test_bf16_mode`.

### Scope changes from original M5 plan

- **Pi 5 8 GB target verified by design, not by hardware.** The streaming
  path is functionally complete and proved correct in the sandbox on
  synthetic-LLaMA inputs; the 60 s / step wall-time target is not measured.
- **AdapterManager + ConversationLogger** were originally bundled with M5;
  AdapterManager landed in the Final pass, ConversationLogger remains a stub
  (its purpose — DPO/KTO data mining — is covered by the toy KTO/DPO tests
  using in-memory data, so no example regression).

### Added — M4 (BPE + mmap dataset + Adafactor)

- **Trainable byte-level BPE tokenizer** (`include/slate/bpe_tokenizer.h`):
  learns merges from a corpus, saves/loads a slate-native vocab file,
  byte-exact round-trip. Smoke test: 276-byte corpus -> vocab 320 -> 80
  tokens (3.45x compression), save/load produces identical encoding.
- **mmap-packed dataset** (`include/slate/mmap_dataset.h`): memory-maps
  a flat int32 token file and samples random `(seq, target)` pairs.
  Used by the GPT example to load tokenized corpora larger than RAM.
- **Adafactor optimizer** (`src/optim/adafactor.c`): factored second
  moment (row+col) for 2D weights cuts optimizer state from AdamW's
  2x weights to roughly 0x. Falls back to per-element for 1D params.
  Smoke test: linear regression converges loss 0.495 -> 0.000 in 200 steps.
- **examples/04_gpt2**: full training pipeline with two commands:
  `prepare` (BPE-train + tokenize corpus to int32 file) and `train`
  (mmap + Adafactor + cosine schedule + grad clip). Verified end-to-end
  on a tiny synthetic corpus.
- New tests: `test_bpe`, `test_mmap_adafactor`.

### Scope changes from original M4 plan

- **bf16 path deferred to M5.** Single-precision is fine for the
  478k-param-class models we can train in the sandbox. bf16 matters
  for 7B+ where the memory savings flip the feasibility on edge HW.
- **Gradient checkpointing deferred to M5.** The `stream.h` design
  already provides the hook (`save_for_backward` + selective_keep_mask);
  implementation goes with the streaming runtime.
- **Karpathy nanoGPT loss comparison not run in sandbox.** Requires
  ~24 hours of CPU time which doesn't fit. The pipeline is verified to
  work; running on user-class hardware is straightforward.

### Added — M3 (SIMD + threadpool + MHA + sampling)

- **Real pthread threadpool** (`src/runtime/threadpool.c`) — fixed-size
  worker pool with atomic task dispatch, condvar wake/sleep, abortable
  shutdown. Replaces the M0 single-threaded stub. Process-wide instance
  reachable via `slate_global_pool()`; thread count controlled by
  `SLATE_NUM_THREADS` env var or auto-detected.
- **SIMD matmul** with explicit AVX2 inner kernel + threaded outer loop.
  Compile with `-mavx2 -mfma` to enable. Benchmark (2-core sandbox):
  ~29 GFLOP/s at 1024x1024x1024 (threaded), ~15 GFLOP/s single-thread AVX2,
  ~12 GFLOP/s scalar baseline. The same kernel is reused by `bmm`.
- **Multi-head attention** via two new ops (`slate_op_permute_12` for 4D
  axis swap, `slate_op_bmm` generalized from 3D to N-D), and the
  `slate_module_mh_attention_new` module. Single-head attention from M2
  is kept as-is for tiny models.
- **`slate_op_causal_mask`** generalized from 3D to N-D so 4D MHA attention
  scores work without a special case.
- **Sampling utilities** (`include/slate/sampling.h`): greedy (T=0),
  temperature, top-k, top-p / nucleus, and combinations. RNG state is
  user-owned so generation is reproducible.
- **Benchmark harness** at `benchmarks/bench_matmul.c` measuring GFLOP/s
  across matrix sizes. The first piece of a broader benchmarks/ tree
  that will expand in M4.
- New tests: `test_threadpool`, `test_mha`, `test_sampling`.

### Scope changes from original M3 plan

- **NEON kernels not implemented.** Sandbox is x86_64-only so we can't
  verify. The `#ifdef __AVX2__` guard makes the SIMD path opt-in cleanly;
  adding NEON later is `#ifdef __ARM_NEON` + analogous intrinsics. Holding
  this until M4 when we cross-compile for Raspberry Pi.
- **Linear scaling claim deferred.** Sandbox has 2 cores so we couldn't
  measure scaling to 4/8 threads. The threadpool is structurally fine
  for it.

### Added — M2 (Tiny Shakespeare GPT)

- Seven new operators (all gradient-checked or smoke-tested):
  `silu`, `scale`, `transpose_last2`, `rms_norm`, `embedding`, `bmm`,
  `causal_mask`. Plus `linear3d` (3D-aware linear projection) used by the
  transformer modules.
- Six new modules: `Embedding`, `RMSNorm`, `SingleHeadAttention`,
  `SwiGLU FFN`, `TransformerBlock` (pre-norm), `CausalLM` (full GPT-style
  decoder). All compose cleanly through the existing autograd graph; no
  changes needed in L0-L2.
- Character-level tokenizer (`slate_char_tokenizer_*`) builds vocab from a
  text corpus and round-trips byte streams to token ids.
- Two new examples: `03_synth_seq` (verifies transformer end-to-end on a
  4-token periodic sequence; 21k params, converges to loss < 1e-4 in 50
  steps) and `03_tinyshakespeare` (1M-param char-level GPT, awaits the
  Karpathy corpus to run).
- `test_transformer_block` end-to-end smoke + finite-difference check on
  input gradient (< 1% relative error).

### Scope changes from original M2 plan

- **Multi-head attention deferred to M3.** M2 ships single-head attention,
  which is sufficient for the 1M-param target. Multi-head requires a
  permute op and 4D `bmm`; bundling with SIMD in M3 is more efficient.
- **RoPE deferred to M4.** We use a learned positional embedding here.
  For tinyshakespeare-class problems this is fine; RoPE matters when we
  start loading LLaMA-format weights in M4.

### Added — M1 (MNIST MLP)

- New operators: `slate_op_softmax`, `slate_op_cross_entropy_loss`
  (fused log_softmax + NLL), `slate_op_add_bias`. All three pass gradient check
  with relative error below 1e-3.
- `slate_lr_scheduler_t` with constant and cosine-with-warmup variants;
  `slate_optimizer_set_lr()` hook added to the optimizer vtable.
- Global-norm gradient clipping (`slate_clip_grad_norm`).
- `slate_idx_*` parser for the MNIST IDX file format.
- `slate_simple_dataloader_t`: synchronous in-memory shuffle + batch.
  (The async prefetch-thread version called out in the ROADMAP M1 entry is
  deferred to M4 — for the in-memory datasets M1 targets, synchronous is
  sufficient and any async would just add complexity without payoff.)
- Two new examples: `02_synth_cls` (a 4-class XOR-like problem verifiable
  in any environment without external data) and `02_mnist` (real MNIST,
  requires user to provide the four IDX files).
- Gradient check coverage extended to seven operators total.

### Fixed — M1

- `slate_module_linear_new(with_bias=true)` now works correctly. The previous
  M0 implementation aliased the bias gradient buffer to a broadcasted view,
  which corrupted memory in any model that used bias. Replaced with a proper
  `add_bias` op whose backward sums correctly into the bias parameter.
- `cross_entropy` backward now owns a deep copy of the targets array instead
  of aliasing the input tensor's data pointer, removing a lifetime hazard
  where the targets tensor going out of scope between forward and backward
  would cause a use-after-free.

### Added — M0 (Autograd + XOR)

- Initial project scaffolding: license, readme, contributing guide, code of
  conduct, security policy.
- ARCHITECTURE.md capturing the full L0–L10 layered design.
- ROADMAP.md with milestones M0–M7 and acceptance criteria.
- CMake-based build with sanitizer support, architecture autodetection, and
  install rules.
- Arena allocator, tensor, static autograd graph, five ops (`matmul`, `add`,
  `mul`, `sigmoid`, `relu`, `mse_loss`), `Linear` / `Sequential` modules,
  SGD and AdamW, and an XOR training example.
- Finite-difference gradient check harness.
- Public-API headers for layers L7–L10 (streaming, teacher, objective, reward,
  executor, mode controller, adapter, data, tokenizer) with full interface
  designs.

## [0.1.0] — unreleased

The first tagged alpha will land after M0 acceptance criteria are met on Linux,
macOS, and Windows in CI.
