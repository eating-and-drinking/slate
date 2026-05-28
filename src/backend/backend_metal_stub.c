// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// backend_metal_stub.c — Metal backend stub.  Returns NULL because
// slate is built without Metal in this configuration.
//
// To enable Metal, replace this file in the build with a
// backend_metal.m (Objective-C++) compiled with -framework Metal on
// macOS.  The vtable contract is the same as the CUDA backend (see
// backend_cuda_stub.c).  Implementation outline:
//
//   * Use MTLDevice = MTLCreateSystemDefaultDevice().
//   * Allocate device buffers via [device newBufferWithLength:options:].
//   * Compile MSL kernels as a single .metallib at build time and load
//     once via [device newLibraryWithSource:].
//   * Encode each compute kernel into a per-call command buffer; the
//     vtable's `sync()` waits on the last commit.

#include "slate/backend.h"

const slate_backend_t* slate_backend_metal(void) { return NULL; }
