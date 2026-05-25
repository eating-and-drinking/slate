# Slate

**A pure C/C++ framework for training and fine-tuning language models on consumer hardware.**

[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![C Standard](https://img.shields.io/badge/C-11-blue.svg)]()
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)]()
[![Milestones](https://img.shields.io/badge/milestones-M0--M7%20done*-green.svg)]()
[![GitHub](https://img.shields.io/badge/github-eating--and--drinking%2Fslate-181717.svg?logo=github)](https://github.com/eating-and-drinking/slate)

Repository: <https://github.com/eating-and-drinking/slate>
Maintainer: [@eating-and-drinking](https://github.com/eating-and-drinking)

Slate is a small, opinionated training framework written in portable C. It targets a
single, specific gap left by `ggml` / `llama.cpp`: **gradient computation, optimizer
state, and a training loop that scales down to laptops, phones, and single-board
computers** — without giving up the streaming, quantization, and edge-friendliness
that made inference on those devices possible.

> Slate is **not** another general-purpose deep-learning framework. It is a focused
> research codebase whose goal is to make local LLM **fine-tuning** (and from-scratch
> training of small models) practical on hardware you already own.

---

## Table of contents

- [What Slate is good at](#what-slate-is-good-at)
- [What Slate is not](#what-slate-is-not)
- [Project status](#project-status)
- [Quick start](#quick-start)
- [Architecture in one diagram](#architecture-in-one-diagram)
- [Realistic capability matrix](#realistic-capability-matrix)
- [Documentation map](#documentation-map)
- [Building](#building)
- [Running the M0 example](#running-the-m0-example)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## What Slate is good at

- **LoRA / QLoRA fine-tuning of mid-sized models** (7B–13B) on laptops with 16 GB+ RAM,
  by streaming base-model weights through a small in-memory working set
- **From-scratch training of small models** (≤500M parameters) on any modern CPU
- **A clean, layered C codebase** small enough to be read end-to-end in a weekend
- **Multiple training objectives**: SFT, knowledge distillation (off- and on-policy),
  DPO, KTO, and GRPO with current bias-corrections (Dr. GRPO + DAPO)
- **Day-time inference plus background training** on the same machine, with explicit
  hardware-protection guardrails (write budgets, thermal limits, battery-aware
  scheduling)
- **Sandboxed code execution** for reinforcement learning on coding tasks, with no
  trust placed in model-generated programs

## What Slate is not

- **Not a PyTorch replacement.** It does not aim for API compatibility, has no Python
  binding by default, and supports a narrow set of model architectures (decoder-only
  transformers in the LLaMA / Qwen / Mistral family).
- **Not a from-scratch training system for ≥7B models on edge hardware.** That is
  physically infeasible on the timescale of a human life — we LoRA-fine-tune
  pre-quantized weights instead.
- **Not production-ready.** This is alpha-quality research software. APIs will change.

## Project status

Slate has reached **milestone M7** (algorithmically complete) plus a Final pass for
engineering completeness. The codebase is roughly **8.4k LOC** under `src/` and
`include/`, plus ~1.6k LOC of examples and ~1.6k LOC of tests.

| Milestone | Topic                                            | Status   |
|-----------|--------------------------------------------------|----------|
| M0        | Autograd + XOR                                   | done     |
| M1        | MNIST MLP, LR scheduler, grad clip               | done\*   |
| M2        | Tiny Shakespeare GPT (single-head)               | done\*   |
| M3        | SIMD + threadpool + MHA + sampling               | done\*   |
| M4        | BPE + mmap dataset + Adafactor + GPT example     | done\*   |
| M5        | LoRA + streaming + GGUF + Q8\_0/Q4\_0 + bf16     | done\*   |
| M6        | TrainingObjective: DPO, KTO, KD, grad-ckpt       | done\*   |
| M7        | GRPO (vanilla/Dr./DAPO), sandbox, code RL        | done\*   |
| Final     | AdapterManager, TeacherCache, NEON, PRM, HTTP    | done     |
| Perf      | Packed-panel GEMM, 8×8 AVX2 µkernel, TLS scratch | done     |

`done*` means the implementation is complete and verified end-to-end in the
sandbox on toy/synthetic inputs, but the *production* acceptance target
(real LLaMA-7B, 32 GB laptop wall time, HumanEval improvement, multi-night
drift comparison) was not run because it requires hardware and corpora
outside the development sandbox. Every `done*` line has a concrete in-tree
test or example that demonstrates correctness.

See [`ROADMAP.md`](ROADMAP.md) for per-milestone acceptance criteria status
and [`CHANGELOG.md`](CHANGELOG.md) for what shipped in each milestone.

## Quick start

```bash
git clone https://github.com/eating-and-drinking/slate.git
cd slate
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/examples/slate_xor             # M0 sanity check
./build/examples/slate_final_assembly  # M7 capstone: GGUF→Q8_0→LoRA→ckpt→save
```

The capstone prints (among other things):

```
[fa] GGUF: 4 tensors
[fa] LoRA trainable params: 2048 (across 8 tensors)
[fa] vs base if trainable: 16384 (1/8.0x compression)
[fa] adapter byte-identical after disk round-trip: yes
[fa] GGUF base file unchanged (byte-level): yes
final_assembly: OK
```

That single example exercises the full chain: GGUF v3 reader → mmap →
Q8\_0 dequant → LoRA wrap (`A` normal-init, `B` zero-init, base frozen) →
gradient checkpointing → AdamW step → `AdapterManager` save → byte-level
re-read confirming both the adapter and the underlying GGUF file are
untouched/correct.

## Architecture in one diagram

Slate is organized as an eight-layer stack. Each layer depends only on the layers
below it. Cross-layer dependencies are a smell.

```
┌─────────────────────────────────────────────────────────────────┐
│  L10  TrainingObjective │ Teacher │ Reward │ CodeExecutor │     │  ← high-level
│  L9   ConversationLog   │ DataLoader │ PreferencePairBuilder    │     algorithms
│  L8   ModeController    │ AdapterManager                        │
├─────────────────────────────────────────────────────────────────┤
│  L7   CLI │ Tokenizer │ Checkpoint │ Logger                     │  ← tooling
├─────────────────────────────────────────────────────────────────┤
│  L6   Trainer │ StreamUnit │ RuntimeMode state machine          │  ← training core
│  L5   Optimizer (SGD, AdamW, Adafactor)                         │
│  L4   Module (Linear, Attention, FFN, TransformerBlock, GPT)    │
├─────────────────────────────────────────────────────────────────┤
│  L3   Operator kernels (forward + backward pairs)               │  ← compute kernels
│  L2   Autograd (graph build, reverse-mode backward)             │
│  L1   Tensor (shape, stride, dtype, grad pointer)               │
├─────────────────────────────────────────────────────────────────┤
│  L0   Runtime (arena allocator, threadpool, SIMD abstraction)   │  ← systems
└─────────────────────────────────────────────────────────────────┘
```

For a complete architectural treatment with rationale, see
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Realistic capability matrix

| Target hardware            | Tiny (≤10M) from scratch | GPT-2 124M from scratch | LoRA 7B fine-tune | LoRA 13B fine-tune |
|----------------------------|:------------------------:|:-----------------------:|:------------------:|:------------------:|
| 32 GB laptop               | hours                    | 1–4 weeks               | hours per epoch    | days per epoch     |
| 16 GB laptop               | hours                    | not recommended         | hours per epoch    | streaming required |
| 8 GB Raspberry Pi 5 + SSD  | day                      | infeasible              | streaming required | infeasible         |
| 8 GB phone (UFS 3.1)       | day                      | infeasible              | streaming required | infeasible         |
| 4 GB phone / Pi Zero       | inference only           | infeasible              | infeasible         | infeasible         |
| microSD-only storage       | **not supported**        | **not supported**       | **not supported**  | **not supported**  |

Anything claiming you can train a 7B model from scratch on a phone in a reasonable
amount of time is selling you something. We're trying not to.

## Documentation map

| Document                                       | What it covers                                              |
|------------------------------------------------|-------------------------------------------------------------|
| [`ARCHITECTURE.md`](ARCHITECTURE.md)           | The complete layered L0–L10 design, rationale per decision  |
| [`ROADMAP.md`](ROADMAP.md)                     | Milestones M0 → M7 with delivered / acceptance status       |
| [`CHANGELOG.md`](CHANGELOG.md)                 | What shipped in each milestone, plus scope changes          |
| [`CONTRIBUTING.md`](CONTRIBUTING.md)           | How to file issues, propose changes, run the test suite     |
| [`SECURITY.md`](SECURITY.md)                   | Reporting security issues; sandbox guarantees               |

The detailed-design notes (streaming, autograd, teacher, objectives, code RL,
safety) are inlined in [`ARCHITECTURE.md`](ARCHITECTURE.md) rather than split
into separate files. The relevant header in `include/slate/` is the canonical
source for each subsystem's interface.

## Building

### Requirements

- C11-capable compiler (gcc ≥ 9, clang ≥ 10, MSVC 2019+)
- CMake ≥ 3.16
- POSIX threads or Win32 threads (provided automatically by CMake)
- No third-party C/C++ dependencies for the core
- Optional: clang-format 14+ for `make format`

### Build types

```bash
# Standard release build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Debug build with assertions and address sanitizer
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DSLATE_SANITIZE=address
cmake --build build-debug -j

# Cross-compile for ARM (example: Raspberry Pi 5)
cmake -S . -B build-arm \
    -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-linux-gnu.cmake \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-arm -j
```

### Build options

| Option              | Default | Description                                              |
|---------------------|---------|----------------------------------------------------------|
| `SLATE_BUILD_TESTS`     | `ON`  | Build the unit-test suite                                |
| `SLATE_BUILD_EXAMPLES`  | `ON`  | Build the example programs                               |
| `SLATE_SANITIZE`        | `""`  | `address`, `undefined`, `thread`, or empty              |
| `SLATE_ENABLE_AVX2`     | `ON`  | Use AVX2 kernels when available (x86_64 only)            |
| `SLATE_ENABLE_NEON`     | `ON`  | Use NEON kernels when available (aarch64 only)           |
| `SLATE_ENABLE_OPENMP`   | `OFF` | Use OpenMP instead of the built-in threadpool            |

## Examples

Each example builds to `build/examples/slate_<name>` and is self-contained.
The "needs corpus" column says whether the example will work in a fresh
checkout or requires the user to supply data first.

| Example                       | What it shows                                       | Needs corpus |
|-------------------------------|-----------------------------------------------------|:------------:|
| `01_xor`                      | M0 autograd + AdamW on XOR                          | no           |
| `02_synth_cls`                | M1 softmax/cross-entropy on a 4-class problem       | no           |
| `02_mnist`                    | M1 MLP on real MNIST IDX files                      | yes          |
| `03_synth_seq`                | M2 21k-param transformer on a periodic sequence     | no           |
| `03_tinyshakespeare`          | M2 ~1M-param char-level GPT                         | yes          |
| `04_gpt2`                     | M4 BPE + mmap + Adafactor end-to-end                | yes          |
| `05_lora`                     | M5 LoRA freeze proof (base byte-identical)          | no           |
| `06_streaming`                | M5 mmap-backed streaming, flat peak RAM             | no           |
| `07_quantized_lora`           | M5 Q8\_0 base + LoRA, on-disk file unchanged        | no           |
| `08_code_rl`                  | M7 GRPO + sandbox + reward, toy code RL loop        | no           |
| `09_final_assembly`           | Capstone: GGUF → Q8\_0 → LoRA → ckpt → AdapterMgr   | no           |

To run every operator's analytic backward against finite differences:

```bash
./build/tests/slate_test_gradcheck
```

The full test suite includes `test_threadpool`, `test_mha`, `test_sampling`,
`test_bpe`, `test_mmap_adafactor`, `test_gguf`, `test_quant`,
`test_gguf_q8_matmul`, `test_dpo`, `test_kto_kd`, `test_grpo`,
`test_sandbox`, `test_checkpoint`, `test_bf16_mode`, `test_adapter_cache`,
and `test_prm_http`. Run them all with `ctest --test-dir build`.

## Contributing

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request. In
particular:

- Every new operator must come with a gradient-check test
- Every new layer must respect the layering rule (no upward dependencies)
- We are very conservative about adding third-party dependencies to the core

## License

Slate is licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE) for
the full text. By contributing, you agree to license your contributions under the same
terms.

## Acknowledgements

Slate would not exist without the careful prior work of:

- **`ggml` and `llama.cpp`** by Georgi Gerganov and contributors — for proving that a
  serious LLM runtime can be written in plain C, and for the arena/threadpool patterns
  this project borrows liberally from
- **`nanoGPT`** by Andrej Karpathy — for the canonical reference of a minimal,
  educational transformer training loop
- **`tinygrad`** by George Hotz and contributors — for showing how small an autograd
  engine can be while still doing useful work
- **DeepSpeed ZeRO-Infinity** — for the streaming/offload techniques this project
  adapts for edge devices
- **DeepSeek**, **ByteDance Seed**, **Qwen team**, and the **Sea AI Lab** for the
  GRPO line of work (GRPO, Dr. GRPO, DAPO, GSPO) that makes RL fine-tuning of small
  models feasible without a learned reward model
