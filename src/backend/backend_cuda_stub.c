// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// backend_cuda_stub.c — CUDA backend stub.  Returns NULL because slate
// is built without CUDA in this configuration.
//
// To actually enable CUDA, replace this file in the build with a
// backend_cuda.cu compiled by nvcc.  See docs/GPU_BACKEND.md for the
// exact vtable contract each function must honour and which test cases
// it must pass.  In rough sketch:
//
//   * `alloc/release` -> cudaMalloc / cudaFree
//   * `copy_h2d/d2h`   -> cudaMemcpyAsync + a per-backend stream
//   * `matvec`         -> cuBLAS sgemm with M=N=1 leading dim
//                          (or hand-rolled __global__ kernel for tiny shapes)
//   * `linear_batch`   -> cuBLAS sgemm with M=B
//   * `rmsnorm_row`    -> single-block reduction kernel (D ≤ 4096
//                          fits in one block on every modern GPU)
//   * `silu_mul`       -> trivial element-wise __global__ kernel
//   * `add_inplace`    -> trivial element-wise
//   * `embed_lookup`   -> one row-copy kernel
//   * `attention_step` -> two kernels (q·K^T + softmax + ·V), or
//                          a single fused kernel for small L
//   * `sync`           -> cudaDeviceSynchronize
//
// The `tests/test_backend.c` file exercises the abstraction and is
// the conformance contract: any concrete CUDA implementation must
// reproduce the CPU backend's outputs within 1e-4 relative error on
// every test input.

#include "slate/backend.h"

const slate_backend_t* slate_backend_cuda(void) { return NULL; }
