// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// bpe_tokenizer.h — byte-level BPE tokenizer.
//
// Vocab starts with 256 single-byte tokens; learned merges build composite
// tokens that span sequences of bytes. Compatible in spirit with GPT-2 BPE
// (byte-level, no preprocessing). Trains from a corpus, saves/loads to a
// simple text format.

#ifndef SLATE_BPE_TOKENIZER_H
#define SLATE_BPE_TOKENIZER_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_bpe_tokenizer slate_bpe_tokenizer_t;

// Train a new tokenizer from `text` (n bytes) targeting `target_vocab_size`
// total tokens. Returns NULL on failure. Training is O(target_vocab * n)
// without aggressive optimization; for corpora ~1MB this is fine.
slate_bpe_tokenizer_t* slate_bpe_train(const char* text, size_t n,
                                        int target_vocab_size);

// Persist to a slate-native vocab file. Format: line 1 "SLATE_BPE 1",
// line 2 "vocab N", then N lines "id\thex_bytes", then "merges M", then
// M lines "left_id right_id new_id".
slate_status_t slate_bpe_save(const slate_bpe_tokenizer_t* tk, const char* path);
slate_bpe_tokenizer_t* slate_bpe_load(const char* path);

void slate_bpe_destroy(slate_bpe_tokenizer_t* tk);

int slate_bpe_vocab_size(const slate_bpe_tokenizer_t* tk);

// Encode bytes -> token ids. Returns number of tokens emitted.
int slate_bpe_encode(const slate_bpe_tokenizer_t* tk,
                      const char* text, size_t n,
                      int32_t* out, int max_tokens);

// Decode token ids -> bytes. Returns bytes written.
int slate_bpe_decode(const slate_bpe_tokenizer_t* tk,
                      const int32_t* tokens, int n,
                      char* out, int max_bytes);

#ifdef __cplusplus
}
#endif

#endif
