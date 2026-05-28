# GPU backend: design + status

This document explains slate's compute-backend abstraction and what's
needed to wire in a real GPU (CUDA / Metal / Vulkan) backend.

## What ships today

- **`include/slate/backend.h`** — `slate_backend_t` vtable of 13
  function pointers (alloc, release, copy_h2d, copy_d2h, matvec,
  linear_batch, rmsnorm_row, silu_mul, add_inplace, embed_lookup,
  attention_step, sync) + a `name` string.

- **`src/backend/backend_cpu.c`** — fully-tested CPU implementation
  that wraps slate's existing AVX2 kernels and the packed-panel GEMM.
  Bit-identical or within fp32 ulp to scalar reference on every
  primitive (`tests/test_backend.c`).

- **`src/backend/backend_cuda_stub.c`** — returns NULL.

- **`src/backend/backend_metal_stub.c`** — returns NULL.

- **`slate_backend_default()`** — prefers cuda → metal → cpu; since
  the GPU backends return NULL, today this always picks the CPU
  backend.

## What's NOT in this milestone

slate does NOT include working CUDA or Metal code.  The reason is
operational, not theoretical: the development sandbox has no nvcc, no
NVIDIA GPU, no Metal compiler, no AMD GPU.  Writing CUDA kernels
without the ability to run them produces "looks correct" code that
ships subtle bugs (uncoalesced loads, bank conflicts, fp16 cast
mistakes, race conditions on shared memory).  Rather than ship
untested kernel code as "production", slate publishes the abstraction
layer + clear hooks and expects whoever has GPU hardware to fill in
the kernels with their own conformance testing against
`tests/test_backend.c`.

## Wiring in a real CUDA backend

Replace `src/backend/backend_cuda_stub.c` in the build with a real
`src/backend/backend_cuda.cu` that defines `slate_backend_cuda()`.
Suggested implementation outline:

| vtable entry      | recommended implementation                       |
|-------------------|--------------------------------------------------|
| `alloc/release`   | `cudaMallocAsync` / `cudaFreeAsync` on a per-backend stream |
| `copy_h2d/d2h`    | `cudaMemcpyAsync` on the same stream             |
| `matvec`          | `cublasSgemv` (M, K) — handles M=1 efficiently   |
| `linear_batch`    | `cublasSgemm` (M=B, K, N)                        |
| `rmsnorm_row`     | single-block reduction kernel (D ≤ 4096 fits one block on modern GPUs) |
| `silu_mul`        | trivial element-wise `__global__` kernel        |
| `add_inplace`     | element-wise                                     |
| `embed_lookup`    | single row-copy kernel                           |
| `attention_step`  | two-pass: (1) q·Kᵀ + softmax  (2) score · V, OR a single fused kernel for small L (≤512) |
| `sync`            | `cudaStreamSynchronize` on the backend's stream  |

The conformance contract is `tests/test_backend.c`.  Any CUDA build
must reproduce the CPU outputs within `1e-4` relative error on every
test input (the CPU primitives themselves clear that threshold by
multiple orders of magnitude — see the actual measured drift in the
test output).

## Wiring in a real Metal backend

Same vtable contract.  Implementation sketch (`backend_metal.m` in
Objective-C / Objective-C++, requires `-framework Metal -framework
Foundation`):

- `MTLDevice* device = MTLCreateSystemDefaultDevice();`
- Allocate device buffers via
  `[device newBufferWithLength:bytes options:MTLResourceStorageModeShared]`
- Compile MSL kernels at build time into a `.metallib` shipped with the
  binary; load once via `[device newLibraryWithSource:]`.
- Encode each compute kernel into a per-call command buffer; the
  vtable's `sync()` waits on the last commit via `[cmd waitUntilCompleted]`.

## Future work (in suggested order)

1. ~~**Refactor the inference engine** to call through `slate_backend_t`~~
   **DONE (L9.1)**.  `slate_infer_engine_new_ex(model, ..., backend)`
   accepts an explicit backend; every primitive in `decode_one` and
   `slate_infer_batch_step` now dispatches through the vtable; session
   alloc goes through `backend->alloc`.  `test_infer` still reports
   `L_inf = 0` after the refactor.

2. **CUDA backend (real)** — see table above.  ~600–800 lines of
   CUDA + sgemm cuBLAS hookup.  Expected throughput on a single
   3090: ~20–30× faster than slate's current single-thread CPU
   matmul for the matmul-bound inference path; less for the
   attention loop until Flash Attention lands.

3. **Multi-GPU sharding** — split the model along the FFN hidden dim
   and the n_heads dim, with NCCL all-reduce between layers.  Not in
   scope for the first GPU milestone.

4. **Flash Attention 2** — fused QKᵀ + softmax + V·attention kernel
   with online softmax.  Cuts attention memory from `O(L²)` to
   `O(L)` and is what makes long-context fast.  ~400 lines CUDA.

5. **fp16 / bf16 weights + accum** — match llama.cpp's typical
   deployment configuration (fp16 weights, fp32 accum) for another
   ~2× memory + bandwidth savings.

## Build flags

```bash
# CPU only (default)
cmake -B build

# With CUDA backend (requires nvcc + a real backend_cuda.cu)
cmake -B build-cuda -DSLATE_ENABLE_CUDA=ON

# With Metal backend (macOS only; requires backend_metal.m)
cmake -B build-metal -DSLATE_ENABLE_METAL=ON
```

When the GPU build flag is ON, CMake substitutes
`backend_cuda_stub.c` (or `backend_metal_stub.c`) for the real
`backend_cuda.cu` / `backend_metal.m` in the library's source list,
and links the appropriate runtime (`cublas` / `Metal.framework`).
