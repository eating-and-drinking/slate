# Slate Architecture

> The complete design of the framework. Read this once before writing any code.

This document captures every load-bearing decision in Slate's design and the reasoning
behind it. It is meant to be read end-to-end; the layered structure that follows
should be visible in every commit.

## Table of contents

1. [Project scope and non-goals](#1-project-scope-and-non-goals)
2. [The memory wall and why streaming matters](#2-the-memory-wall-and-why-streaming-matters)
3. [The eight-layer stack](#3-the-eight-layer-stack)
4. [L0 — Runtime](#l0--runtime)
5. [L1 — Tensor](#l1--tensor)
6. [L2 — Autograd](#l2--autograd)
7. [L3 — Operator kernels](#l3--operator-kernels)
8. [L4 — Modules](#l4--modules)
9. [L5 — Optimizers](#l5--optimizers)
10. [L6 — Training core and streaming runtime](#l6--training-core-and-streaming-runtime)
11. [L7 — Tooling](#l7--tooling)
12. [L8 — Mode controller](#l8--mode-controller)
13. [L9 — Data plane](#l9--data-plane)
14. [L10 — High-level algorithms](#l10--high-level-algorithms)
15. [Cross-cutting: hardware protection](#cross-cutting-hardware-protection)
16. [Cross-cutting: on-disk format](#cross-cutting-on-disk-format)
17. [Realistic capability matrix](#realistic-capability-matrix)
18. [Architectural invariants](#architectural-invariants)

---

## 1. Project scope and non-goals

### In scope

- LoRA / QLoRA fine-tuning of 7B–13B base models on consumer hardware
- From-scratch training of small (≤500M) decoder-only transformers on CPU
- Knowledge distillation (off-policy and on-policy) from local or API teachers
- RLHF without a learned reward model: DPO, KTO, GRPO with current bias-corrections
  (Dr. GRPO, DAPO)
- Reinforcement learning on coding tasks with sandboxed code execution
- Background ("nightly") training alongside foreground inference on the same machine

### Explicitly out of scope

- **From-scratch training of ≥7B models.** Not slow — physically infeasible on
  consumer hardware in any reasonable amount of time. See
  [`docs/design/why-not-from-scratch.md`](docs/design/why-not-from-scratch.md).
- **GPU acceleration as a primary path.** CUDA support may exist as an optional
  backend, but the project's defining constraint is CPU-and-edge correctness.
- **Encoder-only or encoder-decoder architectures.** BERT, T5, BART are not targets.
- **Multi-machine distributed training.** Single-host only. Cross-machine teacher
  serving is supported (because the data volume is small) but not weight-sharding.
- **PyTorch API compatibility.** Slate's API serves Slate. No `torch.*` shim.
- **A Python frontend.** Python is acceptable in `tools/` for offline data
  preparation; the training loop is pure C.

## 2. The memory wall and why streaming matters

To explain every subsequent design decision in one paragraph: training a
transformer needs roughly `5–8 ×` the memory of inference. For LLaMA-7B that is
60–110 GB. A 16 GB laptop, an 8 GB phone, and an 8 GB Raspberry Pi cannot fit a
flat training state. Slate's central design decision is to **stream weights,
gradients, and optimizer state in and out of RAM at the granularity of a
transformer sub-module** (attention or FFN), keep only activations across a
single block boundary, and recompute the rest via gradient checkpointing.

Compared to the alternatives:

- "Just buy more RAM" — not an option on phones, Pis, or many laptops.
- "Just use a smaller model" — fine for inference; for *training* you also need
  somewhere to keep gradients and optimizer state, so the same wall arises one
  level lower.
- "Just use a GPU" — Slate's defining users do not have one.
- "Just write activations to disk" — strictly worse than recomputing (SSD wear,
  I/O latency, no compute saving).

The structural rest of this document follows from this constraint.

## 3. The eight-layer stack

```
┌─────────────────────────────────────────────────────────────────┐
│  L10  TrainingObjective │ Teacher │ Reward │ CodeExecutor       │
│  L9   DataLoader │ ConversationLog │ PreferencePairBuilder      │
│  L8   ModeController    │ AdapterManager                        │
├─────────────────────────────────────────────────────────────────┤
│  L7   CLI │ Tokenizer │ Checkpoint │ Logger                     │
├─────────────────────────────────────────────────────────────────┤
│  L6   Trainer │ StreamUnit │ RuntimeMode                        │
│  L5   Optimizer (SGD, AdamW, Adafactor)                         │
│  L4   Module (Linear, Attention, FFN, TransformerBlock, GPT)    │
├─────────────────────────────────────────────────────────────────┤
│  L3   Operator kernels (forward + backward pairs)               │
│  L2   Autograd (graph build, reverse-mode backward)             │
│  L1   Tensor (shape, stride, dtype, grad pointer)               │
├─────────────────────────────────────────────────────────────────┤
│  L0   Runtime (arena allocator, threadpool, SIMD abstraction)   │
└─────────────────────────────────────────────────────────────────┘
```

A change at layer `N` may use anything in `L0..L(N-1)` and must not refer to
anything in `L(N+1)..L10`. Cross-layer dependencies are reviewed as design flaws.

## L0 — Runtime

The performance ceiling of the entire framework lives here. Three responsibilities:

### Arena allocator

No allocation happens inside the training loop. A `slate_arena_t` is a contiguous
buffer carved into slices; objects allocated against it share a single `free`. We
maintain at least three arenas:

| Arena         | Lifetime                                 | Typical size  |
|---------------|------------------------------------------|---------------|
| `params`      | Whole training run                       | weights+grads |
| `optim`       | Whole training run                       | optimizer state (if not streamed) |
| `step_scratch`| Reset every `optimizer.step()`           | activations   |

`malloc` / `new` in any hot path is a bug.

### Threadpool

A fixed-size pool with work-stealing. No OpenMP dependency: the threadpool must
work on embedded toolchains that lack OMP support. Worker count is tunable at
runtime and adjusted by the `ModeController` based on thermal state.

### SIMD abstraction

A minimal vector layer:

```
vec_add(c, a, b, n)
vec_mul(c, a, b, n)
vec_fma(c, a, b, x, n)        // c[i] = a[i] * b[i] + x[i]
vec_reduce_sum(a, n)
matmul_tile_kernel(...)        // 8x8 or 16x8 micro-kernel
```

With three backends — `scalar` (always available), `avx2`, `neon` — selected at
load time. `matmul` is the only operator where a hand-written micro-kernel pays
off enough to be worth maintaining; everything else can use `vec_*` primitives.

## L1 — Tensor

```c
typedef struct slate_tensor {
    slate_dtype_t  dtype;
    int            n_dims;
    int64_t        shape[SLATE_MAX_DIMS];
    int64_t        stride[SLATE_MAX_DIMS];  // bytes
    void*          data;                    // owned by arena
    void*          grad;                    // optional, owned by arena
    bool           requires_grad;
    struct slate_graph_node* grad_fn;       // L2 hook
    int32_t        ref_count;
} slate_tensor_t;
```

Decisions worth calling out:

- **Stride in bytes, not elements.** Required for handling quantized block dtypes
  uniformly with float dtypes.
- **`grad` is a raw pointer, not a tensor.** Avoids the recursive "gradient of
  the gradient" problem; we only do first-order optimization.
- **Non-contiguous views are first-class.** `permute`, `transpose`, `reshape`
  (when possible) and `slice` produce non-owning views; `contiguous()` copies if
  needed.
- **No implicit dtype promotion.** Kernels accept the dtypes they accept; the
  caller is responsible for casts.

Supported dtypes (initial): `F32`, `F16`, `BF16`, `I32`, `I8`. Quantized block
dtypes (`Q4_0`, `Q8_0`, ggml-compatible) are forward-only and arrive in M5.

## L2 — Autograd

**Static graph, build-once-execute-many.** A `slate_graph_t` records the
operations performed against a `slate_graph_context_t`; forward and backward
both traverse this graph. Static graphs are chosen over PyTorch-style dynamic
graphs because LLM training has almost no data-dependent control flow and the
memory pre-planning a static graph affords is decisive for streaming.

Reverse-mode is implemented as:

1. Topological sort recorded during forward (cheap; just `push_back` to a list).
2. Output node's gradient initialized to 1 (or to upstream loss gradient).
3. Reverse traversal, calling each node's `backward_fn(in_grads, out_grad, ...)`.
4. Gradients accumulate (`+=`) when a tensor feeds multiple consumers.

A gradient context can be invalidated for the inference path: when no
`requires_grad` is set on any input, the graph nodes are not retained,
and intermediate tensors are recycled into `step_scratch`.

## L3 — Operator kernels

This layer is where the majority of code lives. Every operator declares a
forward and a backward kernel:

```c
void slate_op_matmul_forward(slate_tensor_t* out,
                             const slate_tensor_t* a,
                             const slate_tensor_t* b);

void slate_op_matmul_backward(slate_tensor_t* da,
                              slate_tensor_t* db,
                              const slate_tensor_t* a,
                              const slate_tensor_t* b,
                              const slate_tensor_t* d_out);
```

The minimum operator set for LLM training:

| Operator             | Forward                | Backward notes                                  |
|----------------------|------------------------|-------------------------------------------------|
| `matmul`             | tile micro-kernel      | two transposed matmuls                          |
| `add`                | element-wise           | broadcast inverse = sum over broadcast dims     |
| `mul`                | element-wise           | product rule                                    |
| `relu`               | element-wise           | mask by `x > 0`                                 |
| `sigmoid`            | element-wise           | `y*(1-y)`                                       |
| `silu` (swish)       | `x · sigmoid(x)`       | save inputs, not outputs                        |
| `gelu`               | tanh approximation     | derivative formula                              |
| `rms_norm`           | `x / rms(x) · w`       | closed-form, derived in `docs/design/autograd`  |
| `layer_norm`         | mean+var normalize     | standard formula                                |
| `softmax`            | exp + normalize        | Jacobian: `softmax · (d - sum)`                 |
| `cross_entropy`      | fused log_softmax + NLL| `softmax - one_hot` after fusing                |
| `embedding`          | gather                 | scatter-add into gradient                       |
| `rope`               | rotate Q/K             | reverse rotation                                |
| `attention`          | QK^T → softmax → V     | three-link chain rule, fusion-friendly          |

**Rule of thumb**: if two adjacent operators can share state, fuse them. The
`log_softmax + NLL → cross_entropy` fusion is the canonical example; its
backward gradient is the clean `softmax(x) - one_hot(target)`.

Every operator must come with a gradient-check entry in `tests/test_gradcheck.c`
that verifies the analytic backward against a centered finite-difference
estimate.

## L4 — Modules

A C struct-of-function-pointers analogue to `torch.nn.Module`. A `Module`
declares:

- `forward(ctx, x) -> y` — the forward computation.
- `register_params(ps)` — enumerate its trainable parameters for the optimizer
  and the checkpointer.
- (optionally) `register_buffers(bs)` — non-trainable state like RoPE caches.

Pre-built modules (in dependency order):

```
Linear, Sequential                          [M0]
Embedding, LayerNorm, RMSNorm               [M2]
RoPE, MultiHeadAttention                    [M2]
SwiGLU, GatedFFN                            [M2]
TransformerBlock (pre-norm, LLaMA-shape)    [M2]
CausalLM (block stack + lm_head)            [M2]
LoRAAdapter                                 [M5]
```

`Module` is the boundary between "a thing that does compute" (L3) and "a thing
that holds parameters" (L4). Optimizers operate exclusively on parameter sets
produced by `register_params`.

## L5 — Optimizers

```c
typedef struct slate_optimizer {
    void (*step)  (struct slate_optimizer*, slate_param_set_t* params);
    void (*zero_grad)(struct slate_optimizer*, slate_param_set_t* params);
    /* ... state-introspection hooks for the AdapterManager ... */
} slate_optimizer_t;
```

Implementations:

| Optimizer    | State per param | Use case                                          |
|--------------|------------------|---------------------------------------------------|
| `SGD`        | 0 (or m if momentum) | M0 sanity, never for transformers             |
| `AdamW`      | 2× (`m`, `v`)        | default for laptops with ≥ 32 GB              |
| `Adafactor`  | ~0× (row+col)        | required for 7B+ training on edge devices     |
| `8-bit Adam` | 0.5× (`m`, `v` int8) | mid-budget option                             |

Learning-rate scheduling and gradient clipping are not optimizers; they wrap one.

## L6 — Training core and streaming runtime

The two distinguishing complications at this layer are streaming and the
runtime-mode state machine.

### Streaming at sub-module granularity

For a 32-layer model with attention and FFN sub-modules:

- **64 stream units total** (2 per layer).
- **Attention sub-module ≈ 128 MB** (LLaMA-7B, fp16, no GQA).
- **FFN sub-module ≈ 270 MB** (LLaMA-7B, intermediate=11008).
- **Backward peak memory ≈ 470 MB** (one FFN + recomputed activations).

Sub-module granularity is chosen over block granularity because:

1. It composes naturally with **selective activation checkpointing**: the
   activations to keep (block input, attention output) are precisely the
   sub-module boundaries.
2. It cuts backward-pass peak memory by roughly 5× on LLaMA-7B without changing
   the I/O pattern materially.

Sub-tensor (per-matrix) granularity is rejected because:

1. Attention forward needs Q/K/V simultaneously, so the streaming "win" is
   illusory — peak activation would dominate.
2. Smaller transfers trigger SSD IOPS cost more often, hurting eMMC/microSD
   devices.

### `StreamUnit` interface

```c
typedef struct slate_stream_unit {
    void (*load) (struct slate_stream_unit*, slate_arena_t* weights);
    slate_tensor_t* (*forward)(struct slate_stream_unit*, slate_tensor_t* in,
                               slate_kv_cache_t* kv);
    void (*save_for_backward)(struct slate_stream_unit*, slate_cache_t* cache);
    slate_tensor_t* (*backward)(struct slate_stream_unit*, slate_tensor_t* d_out,
                                slate_cache_t* cache);
    void (*evict)(struct slate_stream_unit*);
} slate_stream_unit_t;
```

`save_for_backward` is the selective-checkpoint hook. The default policy keeps
attention outputs (small, expensive to recompute) and discards FFN intermediates
(large, cheap to recompute).

### `RuntimeMode`

One step of on-policy KD requires the runtime to switch modes mid-step. The
state machine:

```
INFERENCE         — all weights resident in RAM, KV cache active, no grads
TRAINING          — streaming weights, gradients allocated, KV cache off,
                    selective checkpoint on
TEACHER_SCORING   — streaming weights, no grads, no KV cache (scoring a fixed
                    sequence with the teacher model)
```

A `StreamUnit` behaves differently in each mode. The mode is a runtime
property of the trainer, not a compile-time choice.

## L7 — Tooling

User-facing utilities. CLI, tokenizer (BPE in C, GGUF-compatible),
checkpointer (writes/reads GGUF-format adapters), logger (TSV; plot externally).

## L8 — Mode controller

A state machine that decides whether the machine is currently in inference
("day") mode or training ("night") mode, and supervises transitions:

```
INFERENCE  ←→  TRAINING
       ↘    ↗
        SUSPENDED  ← (thermal limit / user activity / low battery)
```

Trigger conditions per platform:

- **Linux/macOS laptop**: AC connected + idle ≥ N minutes + time in window
- **Android**: charging + screen off + battery > 50% + thermal < limit
- **Raspberry Pi**: cron + temp sensor below threshold + sufficient disk space

The controller communicates with the trainer through an abort flag and a
checkpoint queue. It does **not** drive the trainer via RPC; the trainer
polls the flag every N steps and saves a checkpoint when it sees the request.

## L9 — Data plane

`ConversationLogger`, `PreferencePairBuilder`, `DataLoader`, and the various
dataset adapters. The day's conversations land in append-only JSONL; nightly
pre-processing tokenizes them, mines preference signals (thumbs, edits,
regenerations, compares), mixes with the replay buffer, and emits a packed
binary file the trainer can mmap.

`PrivacyFilter` lives here. Examples marked `/forget` never leave the file
system; examples marked `allow_external_teacher: false` are filtered before
the API teacher path sees them.

## L10 — High-level algorithms

### Training objectives

```c
typedef struct slate_objective {
    bool (*needs_pair_data)(void);
    bool (*needs_reference)(void);
    bool (*needs_generation)(void);   // on-policy?
    int  (*n_samples_per_step)(void); // 1 except for GRPO

    slate_tensor_t* (*compute_loss)(struct slate_objective*,
                                    const slate_batch_t* batch,
                                    slate_module_t* student,
                                    slate_module_t* reference,
                                    slate_teacher_t* teacher,
                                    slate_reward_t* reward);
} slate_objective_t;
```

Implementations:

| Objective     | Pair data | Reference | Generation | Samples |
|---------------|-----------|-----------|------------|---------|
| `SFT`         | no        | no        | no         | 1       |
| `KD` off-pol. | no        | optional  | no         | 1       |
| `KD` on-pol.  | no        | optional  | yes        | 1       |
| `DPO`         | yes       | yes       | no         | 1       |
| `KTO`         | no        | yes       | no         | 1       |
| `GRPO+DAPO+Dr.GRPO` | no  | yes       | yes        | K=4-8   |
| `Mixed`       | varies    | varies    | varies     | varies  |

The recommended nightly default is a `Mixed` objective:

```
loss = 0.4 · SFT + 0.3 · KTO + 0.2 · KD(SelfTeacher,on_policy) + 0.1 · KD(LocalTeacher,off)
```

The third term is on-policy self-distillation against the frozen base. It
costs ~5% extra compute and is the principal defense against catastrophic
forgetting / drift; we make it nearly mandatory.

### Teacher abstraction

```c
typedef struct slate_teacher {
    bool (*can_score)(void);                 // returns top-k logits?
    int  (*max_k)(void);
    slate_response_t (*generate)(const slate_prompt_t*);
    slate_topk_logits_t (*score)(const slate_tokens_t*, int k);
} slate_teacher_t;
```

Implementations:

- **`SelfTeacher`** — uses the base model already in the streaming pipeline.
  Near-zero cost.
- **`LocalTeacher`** — wraps the Slate inference runtime running a larger
  model. Can be in-process or accessed over a TCP socket for the
  "desktop-as-teacher / laptop-as-student" deployment.
- **`OpenAITeacher`** — REST + top-k logprobs. White-box KD possible (k ≤ 20).
- **`AnthropicTeacher`** — REST only. `can_score() → false`. Sequence-level KD
  only.

### GRPO and improvements

The default RL implementation is **DAPO + Dr.GRPO**. Both are layered on the
common GRPO loop:

1. For each prompt, generate `K` responses with `temperature ≥ 0.8`.
2. For each response, compute a reward.
3. Compute group-relative advantages.
4. Apply policy-gradient update with KL regularization against the reference.

**Dr.GRPO** corrections:
- Drop std-normalization of advantages (cures difficulty bias)
- Drop sequence-length normalization (cures length bias)

**DAPO** additions:
- *Clip-higher*: asymmetric clipping `[1-ε_low, 1+ε_high]` with `ε_high ≈ 0.28`
- *Dynamic sampling*: skip groups where all rewards are identical
- *Token-level loss*: average over tokens, not over sequences
- *Overlong reward shaping*: soft penalty proportional to length overflow

### Sandboxed code execution

Reinforcement learning on coding tasks requires running model-generated code,
which is untrusted input. The sandbox is mandatory:

| Executor                   | Platform   | Speed | Safety |
|----------------------------|------------|-------|--------|
| `WasmtimePyodideExecutor`  | cross-plat | medium| high   |
| `SubprocessSandboxExecutor`| Linux/mac  | high  | medium-high (seccomp + ulimits) |
| `DockerExecutor`           | dev only   | low   | high   |
| `AndroidIsolatedExecutor`  | Android    | medium| medium |

The default is `WasmtimePyodideExecutor` because it is the only one that
works identically on all four platforms we care about.

**Anti-reward-hacking measures are mandatory**:

- Hidden test sets disjoint from training sets
- Process exit code is the only signal, not stderr scraping
- Strict timeout and memory limits
- KL penalty against the reference model with `β ≥ 0.04`
- Per-problem reward cap of 1.0

## Cross-cutting: hardware protection

Slate runs on user-owned hardware. We do not want to wear it out. Four
mechanisms, in order of importance:

### Write budget

Every `Checkpointer` write is metered. A monthly quota (default 50 GB) is
enforced. The streaming design naturally keeps writes per night below 5 GB,
so the budget is a guardrail against misconfiguration, not an everyday
constraint.

### Thermal throttle

The `ModeController` polls system thermal state every 30 seconds.

- Above the soft limit (laptop 75 °C, phone 38 °C surface): halve the
  threadpool worker count
- Above the hard limit (laptop 80 °C, phone 42 °C): pause and save
- Below the resume threshold: resume

### Power policy

Training runs only when AC is connected (laptop) or charging is in progress
(phone). On Android we additionally require the battery temperature is below
the BMS warning threshold. We do **not** override OS-level "optimized battery
charging" — the CLI prompts the user once to enable it.

### Media policy

Training to microSD is blocked by default. The user can override with
`--unsafe-storage` but the CLI warns about projected device lifetime.

## Cross-cutting: on-disk format

The mmap-friendly layout:

```
$ROOT/
├── base/                     # frozen base model
│   ├── manifest.json
│   ├── block_00_attn.bin
│   ├── block_00_ffn.bin
│   ├── ...
│   └── block_31_ffn.bin
├── adapters/
│   ├── current.lora          # promoted, used by inference
│   ├── training.lora.tmp     # tonight's in-progress
│   └── archive/
│       ├── 2026-05-24.lora
│       └── ...
├── data/
│   ├── conversations.jsonl   # day's log
│   ├── replay/               # held-out general samples
│   └── eval/                 # benchmark set
└── state/
    ├── mode                  # current state-machine state
    └── train_checkpoint/     # bit-exact resume state
```

Each `block_*.bin` is a single mmap file with header:

```
[magic:8][version:4][dtype:4][n_tensors:4][reserved:12]
[tensor_table: n_tensors × (name:32, shape:64, offset:8, size:8, dtype:4)]
[raw_data]
```

Both training and inference mmap the same files; the OS page cache becomes a
shared streaming buffer.

## Realistic capability matrix

| Hardware                  | Tiny (≤10M) | GPT-2 124M | LoRA 7B | LoRA 13B |
|---------------------------|:-----------:|:----------:|:-------:|:--------:|
| 32 GB laptop              | hours       | 1–4 weeks  | hours   | days     |
| 16 GB laptop              | hours       | hard       | hours   | streaming|
| 8 GB Pi 5 + SSD           | day         | infeasible | streaming| infeasible|
| 8 GB phone (UFS)          | day         | infeasible | streaming| infeasible|
| 4 GB phone                | infer only  | infeasible | infeasible| infeasible|
| microSD-only              | not supported across the board                |

## Architectural invariants

These hold throughout the codebase. CI is configured to flag violations.

1. **No layer-N file `#include`s a layer-(N+1)+ header.**
2. **No allocation inside a forward or backward pass.** All allocation goes through
   an arena.
3. **Every operator's backward is gradient-checked against finite differences.**
4. **The `slate_` prefix is the only public symbol namespace.**
5. **The C runtime has zero third-party dependencies.** `tools/` may use Python.
6. **No syscalls inside `op_*` kernels.** Logging, timing, debugging hooks are
   provided by L0 and are no-ops in release builds.
7. **`evict()` is idempotent.** Calling it twice does not crash.
8. **Checkpoints are bit-exact resumable.** RNG, optimizer state, dataloader
   position, and step count are all serialized.
