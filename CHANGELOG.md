# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
