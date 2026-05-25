// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// stream_io.h — on-disk format for streamable weight units.
//
// One file per stream unit. Layout:
//   magic[8]   = "SLTSU\0\0\0"
//   version[4] = uint32 LE, currently 1
//   dtype[4]   = uint32 LE, slate_dtype_t
//   n_dims[4]  = uint32 LE
//   shape[8*n_dims] = int64 LE
//   pad to 64 bytes
//   data[numel * sizeof(dtype)]
//
// Files are mmap'd PROT_READ. Loading is just mmap; eviction is munmap.

#ifndef SLATE_STREAM_IO_H
#define SLATE_STREAM_IO_H

#include "slate/types.h"
#include "slate/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Save a tensor's data to a streamable unit file.
slate_status_t slate_stream_write(const slate_tensor_t* t, const char* path);

// mmap a streamable file. Returns a view tensor whose data points into the
// mapped region. The arena is used only for the tensor metadata struct.
// Caller must call slate_stream_release() to munmap when done.
slate_tensor_t* slate_stream_mmap(slate_arena_t* meta_arena, const char* path);

// Release a tensor created by slate_stream_mmap.
void slate_stream_release(slate_tensor_t* t);

// Return resident RAM cost of a mmap'd tensor (size of mapping in bytes).
size_t slate_stream_resident_bytes(const slate_tensor_t* t);

#ifdef __cplusplus
}
#endif

#endif
