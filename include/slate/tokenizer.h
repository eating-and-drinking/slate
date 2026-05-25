// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// tokenizer.h — BPE tokenizer (M2+).
//
// Compatible with GGUF tokenizer.ggml.* fields for interoperability with
// llama.cpp models.

#ifndef SLATE_TOKENIZER_H
#define SLATE_TOKENIZER_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_tokenizer slate_tokenizer_t;

// Load a tokenizer from a GGUF file. Reads vocabulary, merge rules, and
// special token IDs.
slate_tokenizer_t* slate_tokenizer_load_gguf(const char* gguf_path);

// Convenience: load a tokenizer.json (HuggingFace tokenizers format).
slate_tokenizer_t* slate_tokenizer_load_json(const char* json_path);

void slate_tokenizer_destroy(slate_tokenizer_t* tk);

int slate_tokenizer_vocab_size(const slate_tokenizer_t* tk);
int slate_tokenizer_bos_id(const slate_tokenizer_t* tk);
int slate_tokenizer_eos_id(const slate_tokenizer_t* tk);

// Encode UTF-8 text into token IDs. Writes up to `max_tokens` into `out` and
// returns the actual count, or a negative error code.
int slate_tokenizer_encode(const slate_tokenizer_t* tk,
                            const char* text,
                            int32_t* out,
                            int max_tokens,
                            bool add_bos);

// Decode tokens into UTF-8 text. Returns bytes written, or negative on error.
int slate_tokenizer_decode(const slate_tokenizer_t* tk,
                            const int32_t* tokens,
                            int n_tokens,
                            char* out,
                            int max_bytes);

#ifdef __cplusplus
}
#endif

#endif // SLATE_TOKENIZER_H
