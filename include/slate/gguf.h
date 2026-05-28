// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// gguf.h — minimal GGUF v3 reader.
//
// GGUF is llama.cpp's tensor archive format. A single file holds:
//   - magic + version header
//   - typed key-value metadata (architecture, hyperparams, vocab)
//   - tensor info entries (name, shape, dtype, byte offset)
//   - raw tensor data, aligned to a power-of-two boundary
//
// This loader supports the subset Slate currently needs: opens a GGUF file,
// lists tensors by name, and exposes each tensor's data via mmap.
// Quantized dtypes (Q4_0, Q8_0, ...) are recognized but require dequant
// kernels to use in compute (M5.4 work).

#ifndef SLATE_GGUF_H
#define SLATE_GGUF_H

#include "slate/types.h"
#include "slate/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_gguf slate_gguf_t;

slate_gguf_t* slate_gguf_open(const char* path);
void slate_gguf_close(slate_gguf_t* g);

int slate_gguf_n_tensors(const slate_gguf_t* g);
const char* slate_gguf_tensor_name(const slate_gguf_t* g, int idx);

// Look up a tensor by name. Returns a view tensor whose data points into the
// mmap region. Returns NULL if not found. The returned tensor is owned by the
// gguf_t and must NOT be freed individually.
slate_tensor_t* slate_gguf_get_tensor(slate_arena_t* meta_arena,
                                       slate_gguf_t* g,
                                       const char* name);

// Diagnostic: print the metadata table to stdout.
void slate_gguf_dump(const slate_gguf_t* g);

// Metadata KV access.  Returns 0 on success, < 0 if the key is missing
// or the value's type doesn't match.
int slate_gguf_get_u32(const slate_gguf_t* g, const char* key, uint32_t* out);
int slate_gguf_get_f32(const slate_gguf_t* g, const char* key, float* out);
int slate_gguf_get_str(const slate_gguf_t* g, const char* key, const char** out);

#ifdef __cplusplus
}
#endif

#endif
