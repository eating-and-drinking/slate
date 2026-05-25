// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#ifndef SLATE_CHAR_TOKENIZER_H
#define SLATE_CHAR_TOKENIZER_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_char_tokenizer slate_char_tokenizer_t;

// Build a tokenizer by scanning a text buffer for unique byte values.
slate_char_tokenizer_t* slate_char_tokenizer_build(const char* text, size_t n_bytes);
void slate_char_tokenizer_destroy(slate_char_tokenizer_t* tk);

int slate_char_tokenizer_vocab_size(const slate_char_tokenizer_t* tk);

// Encode a UTF-8 byte string into token ids. Returns number of tokens.
int slate_char_tokenizer_encode(const slate_char_tokenizer_t* tk,
                                 const char* text, size_t n_bytes,
                                 int32_t* out, int max_tokens);

// Decode token ids back into bytes. Returns number of bytes written.
int slate_char_tokenizer_decode(const slate_char_tokenizer_t* tk,
                                 const int32_t* tokens, int n_tokens,
                                 char* out, int max_bytes);

#ifdef __cplusplus
}
#endif

#endif
