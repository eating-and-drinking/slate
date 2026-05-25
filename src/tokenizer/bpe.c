// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// bpe.c — byte-level BPE training + encoding.

#include "slate/bpe_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct merge_rule {
    int left;
    int right;
    int new_id;
} merge_rule_t;

struct slate_bpe_tokenizer {
    // Each token = byte sequence; first 256 are individual bytes.
    // We store per-token byte sequences in a flat buffer + offsets.
    uint8_t* bytes_pool;
    size_t bytes_len;
    size_t bytes_cap;
    size_t* tok_offset;     // [vocab]: offset into bytes_pool
    int* tok_len;           // [vocab]: byte length
    int vocab;
    int cap;

    merge_rule_t* merges;   // [n_merges] in order
    int n_merges;
    int cap_merges;
};

static int tok_add(slate_bpe_tokenizer_t* tk, const uint8_t* bytes, int len) {
    if (tk->vocab >= tk->cap) {
        int nc = tk->cap ? tk->cap * 2 : 512;
        tk->tok_offset = (size_t*)realloc(tk->tok_offset, (size_t)nc * sizeof(size_t));
        tk->tok_len = (int*)realloc(tk->tok_len, (size_t)nc * sizeof(int));
        tk->cap = nc;
    }
    if (tk->bytes_len + (size_t)len > tk->bytes_cap) {
        size_t nc = tk->bytes_cap ? tk->bytes_cap * 2 : 4096;
        while (nc < tk->bytes_len + (size_t)len) nc *= 2;
        tk->bytes_pool = (uint8_t*)realloc(tk->bytes_pool, nc);
        tk->bytes_cap = nc;
    }
    tk->tok_offset[tk->vocab] = tk->bytes_len;
    tk->tok_len[tk->vocab] = len;
    memcpy(tk->bytes_pool + tk->bytes_len, bytes, (size_t)len);
    tk->bytes_len += (size_t)len;
    return tk->vocab++;
}

slate_bpe_tokenizer_t* slate_bpe_train(const char* text, size_t n, int target_vocab) {
    if (target_vocab < 256) target_vocab = 256;
    slate_bpe_tokenizer_t* tk = (slate_bpe_tokenizer_t*)calloc(1, sizeof(*tk));
    // Initial 256 byte-tokens
    for (int b = 0; b < 256; ++b) {
        uint8_t bb = (uint8_t)b;
        tok_add(tk, &bb, 1);
    }
    // Working sequence: list of token IDs.
    int* seq = (int*)malloc(n * sizeof(int));
    int slen = 0;
    for (size_t i = 0; i < n; ++i) seq[slen++] = (int)(unsigned char)text[i];

    // Repeatedly find most-frequent adjacent pair and merge.
    int rounds = target_vocab - 256;
    tk->merges = (merge_rule_t*)malloc((size_t)rounds * sizeof(merge_rule_t));
    tk->cap_merges = rounds;

    // For efficiency we use a fixed-size hash to count pairs per round.
    // Pair encoding: (left << 32) | right (uint64). Hash via open-addressing.
    enum { HASH_CAP = 1 << 17 };  // 128k slots
    uint64_t* keys = (uint64_t*)malloc(HASH_CAP * sizeof(uint64_t));
    int64_t* cnts = (int64_t*)malloc(HASH_CAP * sizeof(int64_t));

    for (int round = 0; round < rounds; ++round) {
        memset(keys, 0xff, HASH_CAP * sizeof(uint64_t));  // sentinel ~0
        memset(cnts, 0, HASH_CAP * sizeof(int64_t));
        // Count pairs
        for (int i = 0; i + 1 < slen; ++i) {
            uint64_t k = ((uint64_t)seq[i] << 32) | (uint32_t)seq[i + 1];
            uint64_t h = (k * 2654435761u) & (HASH_CAP - 1);
            while (keys[h] != (uint64_t)-1 && keys[h] != k) h = (h + 1) & (HASH_CAP - 1);
            keys[h] = k; cnts[h]++;
        }
        // Find max
        int64_t best = 0; uint64_t best_k = 0;
        for (int i = 0; i < HASH_CAP; ++i) {
            if (cnts[i] > best) { best = cnts[i]; best_k = keys[i]; }
        }
        if (best < 2) break;  // no pair worth merging
        int left = (int)(best_k >> 32);
        int right = (int)(best_k & 0xFFFFFFFFu);
        // Create new token = concat(left, right)
        int blen = tk->tok_len[left] + tk->tok_len[right];
        uint8_t* buf = (uint8_t*)malloc((size_t)blen);
        memcpy(buf, tk->bytes_pool + tk->tok_offset[left], (size_t)tk->tok_len[left]);
        memcpy(buf + tk->tok_len[left], tk->bytes_pool + tk->tok_offset[right],
               (size_t)tk->tok_len[right]);
        int new_id = tok_add(tk, buf, blen);
        free(buf);
        tk->merges[tk->n_merges++] = (merge_rule_t){left, right, new_id};
        // Rewrite seq replacing adjacent (left, right) with new_id.
        int w = 0;
        for (int i = 0; i < slen; ) {
            if (i + 1 < slen && seq[i] == left && seq[i + 1] == right) {
                seq[w++] = new_id; i += 2;
            } else {
                seq[w++] = seq[i]; i++;
            }
        }
        slen = w;
    }
    free(keys); free(cnts); free(seq);
    return tk;
}

int slate_bpe_vocab_size(const slate_bpe_tokenizer_t* tk) { return tk ? tk->vocab : 0; }

int slate_bpe_encode(const slate_bpe_tokenizer_t* tk, const char* text, size_t n,
                      int32_t* out, int max_tokens) {
    // Start: every byte is a token.
    int* seq = (int*)malloc(n * sizeof(int));
    int slen = 0;
    for (size_t i = 0; i < n; ++i) seq[slen++] = (int)(unsigned char)text[i];
    // Apply merges in the order they were learned. (Slow but correct.)
    for (int m = 0; m < tk->n_merges; ++m) {
        int L = tk->merges[m].left, R = tk->merges[m].right, NEW = tk->merges[m].new_id;
        int w = 0;
        for (int i = 0; i < slen; ) {
            if (i + 1 < slen && seq[i] == L && seq[i + 1] == R) {
                seq[w++] = NEW; i += 2;
            } else {
                seq[w++] = seq[i]; i++;
            }
        }
        slen = w;
    }
    int copy = slen < max_tokens ? slen : max_tokens;
    for (int i = 0; i < copy; ++i) out[i] = seq[i];
    free(seq);
    return copy;
}

int slate_bpe_decode(const slate_bpe_tokenizer_t* tk, const int32_t* tokens, int n,
                      char* out, int max_bytes) {
    int w = 0;
    for (int i = 0; i < n && w < max_bytes; ++i) {
        int t = tokens[i];
        if (t < 0 || t >= tk->vocab) continue;
        int tl = tk->tok_len[t];
        const uint8_t* tb = tk->bytes_pool + tk->tok_offset[t];
        for (int j = 0; j < tl && w < max_bytes; ++j) out[w++] = (char)tb[j];
    }
    return w;
}

void slate_bpe_destroy(slate_bpe_tokenizer_t* tk) {
    if (!tk) return;
    free(tk->bytes_pool);
    free(tk->tok_offset);
    free(tk->tok_len);
    free(tk->merges);
    free(tk);
}

slate_status_t slate_bpe_save(const slate_bpe_tokenizer_t* tk, const char* path) {
    FILE* fp = fopen(path, "wb"); if (!fp) return SLATE_ERR_IO;
    fprintf(fp, "SLATE_BPE 1\n");
    fprintf(fp, "vocab %d\n", tk->vocab);
    for (int i = 0; i < tk->vocab; ++i) {
        fprintf(fp, "%d ", i);
        const uint8_t* b = tk->bytes_pool + tk->tok_offset[i];
        for (int j = 0; j < tk->tok_len[i]; ++j) fprintf(fp, "%02x", b[j]);
        fprintf(fp, "\n");
    }
    fprintf(fp, "merges %d\n", tk->n_merges);
    for (int i = 0; i < tk->n_merges; ++i) {
        fprintf(fp, "%d %d %d\n", tk->merges[i].left, tk->merges[i].right, tk->merges[i].new_id);
    }
    fclose(fp);
    return SLATE_OK;
}

slate_bpe_tokenizer_t* slate_bpe_load(const char* path) {
    FILE* fp = fopen(path, "rb"); if (!fp) return NULL;
    char hdr[64]; int ver;
    if (fscanf(fp, "%63s %d\n", hdr, &ver) != 2 || strcmp(hdr, "SLATE_BPE") != 0) {
        fclose(fp); return NULL;
    }
    slate_bpe_tokenizer_t* tk = (slate_bpe_tokenizer_t*)calloc(1, sizeof(*tk));
    int V; fscanf(fp, "vocab %d\n", &V);
    for (int i = 0; i < V; ++i) {
        int id; char hex[8192];
        fscanf(fp, "%d %8191s\n", &id, hex);
        int blen = (int)(strlen(hex) / 2);
        uint8_t bytes[8192];
        for (int j = 0; j < blen; ++j) {
            unsigned int v; sscanf(hex + 2 * j, "%2x", &v); bytes[j] = (uint8_t)v;
        }
        tok_add(tk, bytes, blen);
    }
    int M; fscanf(fp, "merges %d\n", &M);
    tk->merges = (merge_rule_t*)malloc((size_t)M * sizeof(merge_rule_t));
    tk->cap_merges = M;
    for (int i = 0; i < M; ++i) {
        fscanf(fp, "%d %d %d\n", &tk->merges[i].left, &tk->merges[i].right, &tk->merges[i].new_id);
    }
    tk->n_merges = M;
    fclose(fp);
    return tk;
}
