// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#include "slate/char_tokenizer.h"
#include <stdlib.h>
#include <string.h>

struct slate_char_tokenizer {
    int byte_to_id[256];  // -1 if not in vocab
    int id_to_byte[256];
    int vocab;
};

slate_char_tokenizer_t* slate_char_tokenizer_build(const char* text, size_t n) {
    slate_char_tokenizer_t* tk = (slate_char_tokenizer_t*)calloc(1, sizeof(*tk));
    for (int i = 0; i < 256; ++i) tk->byte_to_id[i] = -1;
    for (size_t i = 0; i < n; ++i) {
        unsigned char c = (unsigned char)text[i];
        if (tk->byte_to_id[c] < 0) {
            tk->byte_to_id[c] = tk->vocab;
            tk->id_to_byte[tk->vocab] = c;
            tk->vocab++;
        }
    }
    return tk;
}

void slate_char_tokenizer_destroy(slate_char_tokenizer_t* tk) { free(tk); }
int slate_char_tokenizer_vocab_size(const slate_char_tokenizer_t* tk) { return tk ? tk->vocab : 0; }

int slate_char_tokenizer_encode(const slate_char_tokenizer_t* tk,
                                 const char* text, size_t n,
                                 int32_t* out, int max_tokens) {
    int written = 0;
    for (size_t i = 0; i < n && written < max_tokens; ++i) {
        int id = tk->byte_to_id[(unsigned char)text[i]];
        if (id < 0) continue;  // unknown char: skip
        out[written++] = (int32_t)id;
    }
    return written;
}

int slate_char_tokenizer_decode(const slate_char_tokenizer_t* tk,
                                 const int32_t* tokens, int n,
                                 char* out, int max_bytes) {
    int written = 0;
    for (int i = 0; i < n && written < max_bytes; ++i) {
        int id = tokens[i];
        if (id < 0 || id >= tk->vocab) continue;
        out[written++] = (char)tk->id_to_byte[id];
    }
    return written;
}
