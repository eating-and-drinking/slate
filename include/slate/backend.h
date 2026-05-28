// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// backend.h — compute backend abstraction for the inference fast-path.
//
// The CPU inference engine inlines a small set of primitives (matvec,
// RMSNorm, SiLU·mul, embedding lookup, single-token attention).  A
// production-grade serve-from-GPU path needs the same primitives but
// dispatched to CUDA / Metal / Vulkan kernels with device-side memory.
// Rather than #ifdef-soup these into the inference engine itself, this
// header carves the dispatch layer cleanly:
//
//   * `slate_backend_t` is a vtable of function pointers — one entry
//     per primitive the engine needs.
//   * `slate_backend_cpu()` returns a fully-tested CPU implementation
//     that wraps the existing AVX2 kernels.
//   * `slate_backend_cuda()` / `slate_backend_metal()` are STUBS in
//     this layer (return NULL).  Filling them in requires real GPU
//     hardware to test against — see `docs/GPU_BACKEND.md` for the
//     exact contract each function must honour and what's tested vs
//     untested.
//
// Allocation: every backend's `alloc()` returns a *device pointer*
// from the backend's POV.  For the CPU backend that's just a host
// pointer.  For a CUDA backend it's a `cudaMalloc` pointer; you must
// use the backend's `copy_h2d` / `copy_d2h` to move data into/out of
// CPU buffers.  The inference engine, once it's wired through the
// backend (next milestone), will allocate weights + activations
// through the backend so the same code path serves CPU or GPU.

#ifndef SLATE_BACKEND_H
#define SLATE_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_backend {
    // Identification (e.g. "cpu", "cuda", "metal")
    const char* name;

    // ---- Memory management ----
    // alloc/free a buffer on the backend's device.
    void* (*alloc)(size_t bytes);
    void  (*release)(void* dev_ptr);

    // Host <-> device transfer.  For CPU these are plain memcpy.
    void (*copy_h2d)(void* dst_dev, const void* src_host, size_t bytes);
    void (*copy_d2h)(void* dst_host, const void* src_dev, size_t bytes);

    // ---- Inference primitives ----
    // All operate on float pointers that are valid on the backend's device.

    // y[M] = A[M, K] @ x[K]     (single matvec; row-major A)
    void (*matvec)(float* y, const float* A, const float* x, int M, int K);

    // C[B, N] = X[B, K] @ W[K, N]   (batched linear projection — the
    // M=B kernel used by continuous batching)
    void (*linear_batch)(float* C, const float* X, const float* W,
                         int B, int K, int N);

    // y[M] = A_Q4K[M, K] @ x[K]   (Q4_K-quantised weight matvec)
    // A_Q4K is laid out as M rows, each (K/256) Q4_K super-blocks of 144 bytes.
    // K must be a multiple of 256.  Returns y in f32.
    // Optional: NULL on backends that don't yet implement Q4_K (callers
    // must check and fall back to f32 dequant + matvec).
    void (*matvec_q4k)(float* y, const void* A_q4k, const float* x, int M, int K);

    // y[D] = w[d] * x[d] / sqrt(mean(x²) + eps)    (single-token row)
    void (*rmsnorm_row)(float* y, const float* x, const float* w,
                        int D, float eps);

    // y[H] = silu(gate[H]) * up[H]      (SwiGLU activation, in-place
    // on the gate buffer is the typical pattern)
    void (*silu_mul)(float* y, const float* gate, const float* up, int H);

    // y[D] += x[D]
    void (*add_inplace)(float* y, const float* x, int D);

    // y[D] = embedding_W[token_id, :]    (gather a single row)
    void (*embed_lookup)(float* y, const float* W, int token_id, int D);

    // Single-token causal attention against an already-populated KV cache.
    //   q       : [D]    current-position Q
    //   K_cache : [L, D] keys for positions 0..L-1
    //   V_cache : [L, D] values for positions 0..L-1
    //   scores  : [L]    scratch (length must cover the cache)
    //   out     : [D]    result
    // Implements: softmax((q @ K^T) / sqrt(D)) @ V
    void (*attention_step)(float* out, const float* q,
                           const float* K_cache, const float* V_cache,
                           float* scores, int L, int D);

    // Wait for all enqueued work to complete (no-op on CPU; cudaDeviceSynchronize on GPU).
    void (*sync)(void);
} slate_backend_t;

// Backends.  Each returns a pointer to a singleton vtable, or NULL if
// the backend isn't compiled into this build / no compatible hardware
// is present at runtime.
const slate_backend_t* slate_backend_cpu(void);
const slate_backend_t* slate_backend_cuda(void);
const slate_backend_t* slate_backend_metal(void);

// Pick the best available backend (preference: cuda > metal > cpu).
// Always non-NULL (cpu is always available).
const slate_backend_t* slate_backend_default(void);

#ifdef __cplusplus
}
#endif

#endif // SLATE_BACKEND_H
