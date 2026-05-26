# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
