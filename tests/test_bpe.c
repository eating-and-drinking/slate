// SPDX-License-Identifier: Apache-2.0
#include "slate/bpe_tokenizer.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(void) {
    const char* corpus =
        "the quick brown fox jumps over the lazy dog. the quick brown fox "
        "jumps over the lazy dog. the cat sat on the mat. the cat sat on the mat. "
        "to be or not to be, that is the question. to be or not to be. "
        "all happy families are alike; each unhappy family is unhappy in its own way.";
    size_t n = strlen(corpus);
    slate_bpe_tokenizer_t* tk = slate_bpe_train(corpus, n, 320);
    printf("[bpe] vocab=%d\n", slate_bpe_vocab_size(tk));
    int32_t toks[2048];
    int nt = slate_bpe_encode(tk, corpus, n, toks, 2048);
    printf("[bpe] encoded %zu bytes -> %d tokens (ratio %.2fx)\n", n, nt, (double)n / nt);
    char back[4096];
    int nb = slate_bpe_decode(tk, toks, nt, back, sizeof(back));
    int ok = (nb == (int)n && memcmp(back, corpus, n) == 0);
    printf("[bpe] round-trip: %s\n", ok ? "OK" : "FAIL");
    // Save and reload
    slate_bpe_save(tk, "/tmp/slate_bpe_test.vocab");
    slate_bpe_tokenizer_t* tk2 = slate_bpe_load("/tmp/slate_bpe_test.vocab");
    int32_t toks2[2048];
    int nt2 = slate_bpe_encode(tk2, corpus, n, toks2, 2048);
    int eq = (nt == nt2);
    for (int i = 0; i < nt && eq; ++i) if (toks[i] != toks2[i]) eq = 0;
    printf("[bpe] save/load: %s\n", eq ? "OK" : "FAIL");
    ok = ok && eq;
    slate_bpe_destroy(tk); slate_bpe_destroy(tk2);
    printf("test_bpe: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
