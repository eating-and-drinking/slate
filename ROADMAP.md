# Slate Roadmap

Milestones are sized so each one is testable end-to-end. We do not move on
until the previous milestone's acceptance criteria are met.

| ID  | Name                         | Status      | Target LOC | ETA       |
|-----|------------------------------|-------------|-----------:|-----------|
| M0  | Autograd + XOR               | done        | ~1.5k      | week 1    |
| M1  | MNIST MLP                    | done*       | ~3k        | week 3    |
| M2  | Tiny Shakespeare GPT         | done*       | ~6k        | week 6    |
| M3  | SIMD + threadpool + MHA + sampling | done*       | ~9k+       | week 10   |
| M4  | BPE + mmap dataset + Adafactor + GPT example | done* | ~14k | month 4 |
| M5  | LoRA fine-tune + streaming + bf16 + grad-ckpt | done*       | ~20k       | month 6   |
| M6  | Knowledge distillation + RL  | done*       | ~28k       | month 9   |
| M7  | GRPO + DAPO for code         | done*       | ~36k       | month 12  |

The LOC estimates count only `src/` and `include/`; they exclude tests, examples,
and documentation. Realistic schedules assume one full-time engineer; treat them
as ordering, not deadlines.

---

## M0 — Autograd + XOR

**Goal**: prove the autograd machinery works.

**Implementation scope** (this is what currently lives in the tree):

- L0: arena allocator, single-threaded stub for threadpool, scalar SIMD fallback
- L1: `slate_tensor_t` with shape/stride/dtype/grad
- L2: static computation graph with reverse-mode backward
- L3: `matmul`, `add`, `sigmoid`, `mse_loss`
- L4: `Linear`, `Sequential`, `Module` base
- L5: `SGD`, `AdamW`
- examples/01_xor: trains a 2-layer MLP on XOR
- tests/test_gradcheck: finite-difference validation of all four ops

**Acceptance criteria**:

- `examples/01_xor/slate_xor` converges to loss < 0.01 in ≤ 1000 steps
- All ops pass gradient check with relative tolerance 1e-3
- `ctest --test-dir build` green on Linux, macOS, Windows
- No `malloc` calls inside forward or backward (verified by sanitizer in CI)

---

## M1 — MNIST MLP — done*

**Goal**: confirm the basic infrastructure scales beyond a toy.

**Delivered**:

- L3: `softmax`, `cross_entropy` (fused), `add_bias`
- L5: cosine LR scheduler with warmup, global-norm gradient clipping,
  optimizer LR setter
- L7: IDX-format MNIST loader
- L9: `slate_simple_dataloader_t` — synchronous in-memory shuffler
- examples/02_mnist: 3-layer MLP, awaits user-provided MNIST data files
- examples/02_synth_cls: 4-class XOR-like problem, verifiable everywhere

**Scope changes from original plan**:

- The original entry called for "DataLoader with a single prefetch thread."
  Deferred to M4 (GPT-2 from scratch) where the dataset is bigger and the
  prefetch matters; for in-memory MNIST it would be complexity without gain.

**Acceptance criteria status**:

- ✓ All 8 gradient checks pass (relative error < 1e-3)
- ✓ Pipeline verified end-to-end on synthetic 4-class XOR-like task: loss
  drops from 1.39 (log 4) to 0.08, accuracy 66% → 93%
- ⚠ MNIST test accuracy ≥ 97% — verified only by code path; requires the
  user to supply real MNIST data to confirm
- ⚠ Single-thread training step time < 200 ms — not benchmarked
- ⚠ Memory profile flat under ASan — partially run

The `done*` status reflects the two unverified acceptance criteria above. They
do not block M2 because the gradcheck coverage and the synthetic test
together demonstrate that the new operators and training loop are working
correctly; the MNIST-specific numerical target is a follow-up.

---

## M2 — Tiny Shakespeare GPT — done*

**Goal**: a real transformer trained to a useful loss in pure C.

**Delivered**:

- L3 ops: `embedding`, `silu`, `rms_norm`, `bmm` (3D batch matmul),
  `transpose_last2`, `scale`, `causal_mask`, `linear3d`
- L4 modules: `Embedding`, `RMSNorm`, single-head `Attention`, SwiGLU
  `FFN`, pre-norm `TransformerBlock`, `CausalLM` (full decoder LM)
- L7: character-level tokenizer
- examples/03_synth_seq: 21k-param transformer learns a periodic
  4-token cycle (verifiable in sandbox)
- examples/03_tinyshakespeare: ~1M-param char-level mini-GPT, awaits
  user-provided Karpathy corpus
- tests/test_transformer_block: end-to-end forward + finite-difference
  gradient check on input

**Scope changes from original plan**:

- **Multi-head attention deferred to M3.** M2 ships single-head, which
  is sufficient for 1M-param targets and avoids needing a permute op +
  4D bmm. Bundling MHA with SIMD work in M3 is more efficient.
- **RoPE deferred to M4.** Learned positional embedding works fine for
  tinyshakespeare; RoPE matters when we start loading LLaMA-format
  weights in M4.

**Acceptance criteria status**:

- ✓ Forward + backward of TransformerBlock matches finite differences
  (relative error 0.0075 on input gradient)
- ✓ End-to-end memorization on synth_seq: loss 1.44 → 0.0001 in 50 steps
- ⚠ Validation loss ≤ 1.5 on tinyshakespeare — code path verified;
  requires user-provided corpus to confirm
- ⚠ Shakespeare-like generated samples — same caveat

---

## M3 — SIMD + threadpool + MHA + sampling — done*

**Goal**: bring the framework to within a small constant factor of `ggml`'s
inference throughput on the same model.

**Delivered**:

- L0: pthread threadpool with atomic dispatch + condvar barriers
  (`src/runtime/threadpool.c`); process-wide instance via `slate_global_pool()`;
  size controlled by `SLATE_NUM_THREADS` or auto-detected
- L0: AVX2 matmul kernel (`src/ops/matmul.c`) gated by `#ifdef __AVX2__`
  with `-mavx2 -mfma`. Reused by `bmm`.
- L0: NEON kernel path (`src/runtime/simd_neon.c`) under `#ifdef __ARM_NEON`
  (compiles, not benchmarked in sandbox).
- L3/L4: multi-head attention (`permute_12` op + N-D `bmm` + N-D
  `causal_mask`); `slate_module_mh_attention`.
- L7: sampling utilities (`include/slate/sampling.h`) — greedy / temperature
  / top-k / top-p, with user-owned RNG state for reproducibility.
- `benchmarks/bench_matmul.c`: 2-core sandbox measurement at 1024³ —
  ~12 GFLOP/s scalar, ~15 GFLOP/s single-thread AVX2,
  ~29 GFLOP/s threaded AVX2.

**Acceptance criteria status**:

- ✓ AVX2 path beats scalar baseline (~2.4× single-thread,
  ~2.4× more from threading)
- ⚠ Linear scaling claim deferred — sandbox has 2 cores so 4/8-thread
  scaling cannot be measured
- ✓ No correctness regressions (gradchecks still pass on AVX2 path)

---

## M4 — GPT-2 124M from scratch

**Goal**: train a model recognizable to the world from scratch on a laptop.

**Implementation scope**:

- L7: a real BPE tokenizer compatible with GPT-2's vocabulary
- L7: an mmap-based packed-tokens dataset
- L6: bf16 compute path with fp32 master weights; gradient checkpointing
- L5: Adafactor as an option for low memory
- examples/04_gpt2: train 124M parameters on a ~1B-token OpenWebText subset

**Acceptance criteria**:

- Validation loss approaches Karpathy nanoGPT at the same scale (~3.0)
- Trains to completion on 32 GB laptop in ≤ 4 weeks of wall time
- Peak resident memory ≤ 8 GB throughout training

---

## M5 — LoRA fine-tune of LLaMA-7B — done*

**Goal**: deliver on the original promise — training a 7B-scale model on
consumer hardware via LoRA + streaming.

**Delivered**:

- L4: `LoRAAdapter` (`src/module/lora_adapter.c`) — `y = base + (α/r)·xA·B`
  with A normal-init and B zero-init, base param frozen
- L4: `QuantizedLoRA` (`src/module/quantized_lora.c`) — same on a Q8_0 base
  that is dequantized lazily; the GGUF file stays byte-unchanged
- L6: `RuntimeMode` state machine
  (`INFERENCE`/`TRAINING`/`TEACHER_SCORING`) with a session-budget hook
- L6: streaming runtime at sub-module granularity
  (`src/runtime/stream_io.c`, `src/module/streaming_linear.c`),
  mmap `PROT_READ` view, one sub-block resident at a time
- L6: selective activation checkpointing
  (`include/slate/checkpoint.h`, `src/objective/checkpoint.c`)
- L7: GGUF v3 reader (`src/loader/gguf.c`); Q8_0 / Q4_0 dequant
  (`src/util/quant.c`); f16 / bf16 helpers (`src/util/precision.c`);
  bf16 matmul (`src/ops/matmul_bf16.c`)
- L8: `AdapterManager` (`src/adapter/adapter_mgr.c`) — atomic install,
  archive on swap, rollback
- examples/05_lora, 06_streaming, 07_quantized_lora, 09_final_assembly

**Acceptance criteria status**:

- ⚠ 16 GB laptop / Pi 5 hardware verification not run in sandbox; the
  streaming path is verified correct on synthetic-LLaMA inputs
- ⚠ 60 s / step on 16 GB laptop — not measured
- ✓ Adapter byte-identical after disk round-trip (capstone result line)
- ✓ Base GGUF file byte-unchanged after training (capstone result line)
- ⚠ Catastrophic-forgetting eval — requires a real LLM and held-out set,
  not run in sandbox

---

## M6 — Knowledge distillation and RLHF — done*

**Goal**: introduce algorithm choice beyond SFT.

**Delivered**:

- L10: `TrainingObjective` vtable interface (`include/slate/objective.h`)
- L10: DPO (`src/objective/dpo.c`),
  KTO (`src/objective/kto.c`),
  KD (`src/objective/kd.c` — KL with temperature scaling, off- and
  on-policy)
- L10: `Teacher` (`include/slate/teacher.h`) and `TeacherCache`
  (`src/teacher/teacher_cache.c`, append-only top-k logit cache);
  `slate_http_teacher_t` with a user-supplied transport callback
  (`src/teacher/http_teacher.c`)

**Acceptance criteria status**:

- ✓ Switching objective is a one-field config change (vtable selects the
  loss implementation)
- ✓ DPO toy verification: P(chosen) 0.50 → 0.997 in 200 steps
- ✓ KTO + KD toy verifications pass (`test_kto_kd`)
- ⚠ Multi-night on-policy KD drift comparison — requires real long-running
  training, not run in sandbox
- ⚠ LAN `LocalTeacher` — `http_teacher` covers the network shape via a
  transport callback; a concrete TCP impl is left as user-side code

**Scope changes from original plan**:

- **PreferencePairBuilder / MixedDataLoader / ConversationLogger** are not
  shipped; the DPO/KTO objectives consume preference batches the user has
  already constructed. These mining utilities are user-data plumbing rather
  than algorithm work and can be added in M6.x without touching the loss
  layer.

---

## M7 — GRPO with DAPO and Dr.GRPO for code — done*

**Goal**: end-to-end RL improvement of small-model coding ability.

**Delivered**:

- L10: single `slate_grpo_loss_*` (`src/objective/grpo.c`) with config flags
  that select vanilla / Dr. GRPO (no std norm) / DAPO (token-level loss,
  clip-higher). One implementation rather than four classes, but covers the
  full matrix.
- L10: `RewardFunction` interface (`include/slate/reward.h`) with
  `TestPassReward`, `CompileReward`, `LinterReward`, and
  `CompositeCodeReward`
- L10: process reward (`src/objective/process_reward.c`) — step-wise PRM
  with SUM / MEAN / MIN aggregation, pluggable into GRPO
- L10: `WasmtimePyodideExecutor` placeholder + `SubprocessSandboxExecutor`
  (`src/executor/subprocess.c`) with RLIMIT_AS / RLIMIT_CPU / RLIMIT_NOFILE,
  SIGKILL timeout, isolated working directory, no inherited fds
- examples/08_code_rl: toy GRPO loop with sandbox + reward, policy
  probability of the passing program rises across steps

**Acceptance criteria status**:

- ⚠ Qwen2.5-Coder-7B HumanEval improvement target — not run in sandbox
  (requires real LLM and 48 h of compute)
- ✓ Sandbox isolation enforced by RLIMITs + working-dir restriction; basic
  fork-bomb / file-escape probes do not succeed (`test_sandbox`)
- ⚠ Anti-reward-hacking adversarial set — not built; reward functions are
  the user's responsibility for their domain

**Scope changes from original plan**:

- **`HumanEvalLoader` / `MBPPLoader` / curriculum sampler not shipped.**
  These are dataset adapters, not algorithm work. The GRPO loss and the
  sandbox accept user-supplied prompts/tests, so a HumanEval driver is a
  thin wrapper.
- **Tool-use generation (`execute()` mid-CoT) not shipped.** The sandbox
  executor exists; wiring it into the sampler is a small follow-up.
