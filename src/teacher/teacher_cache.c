// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
#define _POSIX_C_SOURCE 200809L
#include "slate/teacher_cache.h"
#include "slate/error.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct entry { int32_t prompt_id; long file_offset; int seq_len, k; } entry_t;

struct slate_teacher_cache {
    char* path; FILE* fp;
    entry_t* entries; int n_entries, cap_entries;
};

static void rebuild_index(slate_teacher_cache_t* c) {
    fseek(c->fp, 0, SEEK_SET);
    int32_t hdr[3];
    while (fread(hdr, 4, 3, c->fp) == 3) {
        long pos = ftell(c->fp);
        if (c->n_entries >= c->cap_entries) {
            int nc = c->cap_entries ? c->cap_entries * 2 : 16;
            c->entries = (entry_t*)realloc(c->entries, (size_t)nc * sizeof(entry_t));
            c->cap_entries = nc;
        }
        c->entries[c->n_entries++] = (entry_t){hdr[0], pos, hdr[1], hdr[2]};
        long payload = (long)hdr[1] * hdr[2] * 8;
        fseek(c->fp, payload, SEEK_CUR);
    }
    clearerr(c->fp);
}

slate_teacher_cache_t* slate_teacher_cache_open(const char* path) {
    slate_teacher_cache_t* c = (slate_teacher_cache_t*)calloc(1, sizeof(*c));
    c->path = strdup(path);
    c->fp = fopen(path, "a+b");
    if (!c->fp) { free(c->path); free(c); return NULL; }
    rebuild_index(c);
    return c;
}
void slate_teacher_cache_close(slate_teacher_cache_t* c) {
    if (!c) return;
    fclose(c->fp); free(c->entries); free(c->path); free(c);
}
slate_status_t slate_teacher_cache_put(slate_teacher_cache_t* c, int32_t pid,
                                        int seq_len, int k,
                                        const int32_t* tokens, const float* logits) {
    if (!c) return SLATE_ERR_INVALID_ARGUMENT;
    fseek(c->fp, 0, SEEK_END);
    int32_t hdr[3] = {pid, seq_len, k};
    if (fwrite(hdr, 4, 3, c->fp) != 3) return SLATE_ERR_IO;
    long payload_off = ftell(c->fp);
    int n = seq_len * k;
    for (int i = 0; i < n; ++i) {
        fwrite(&tokens[i], 4, 1, c->fp);
        fwrite(&logits[i], 4, 1, c->fp);
    }
    fflush(c->fp);
    if (c->n_entries >= c->cap_entries) {
        int nc = c->cap_entries ? c->cap_entries * 2 : 16;
        c->entries = (entry_t*)realloc(c->entries, (size_t)nc * sizeof(entry_t));
        c->cap_entries = nc;
    }
    c->entries[c->n_entries++] = (entry_t){pid, payload_off, seq_len, k};
    return SLATE_OK;
}
slate_status_t slate_teacher_cache_get(slate_teacher_cache_t* c, int32_t pid,
                                        int* out_seq, int* out_k,
                                        int32_t** out_toks, float** out_logits) {
    for (int i = 0; i < c->n_entries; ++i) {
        if (c->entries[i].prompt_id == pid) {
            entry_t* e = &c->entries[i];
            *out_seq = e->seq_len; *out_k = e->k;
            int n = e->seq_len * e->k;
            int32_t* t = (int32_t*)malloc((size_t)n * sizeof(int32_t));
            float* l = (float*)malloc((size_t)n * sizeof(float));
            fseek(c->fp, e->file_offset, SEEK_SET);
            for (int j = 0; j < n; ++j) { fread(&t[j], 4, 1, c->fp); fread(&l[j], 4, 1, c->fp); }
            *out_toks = t; *out_logits = l;
            return SLATE_OK;
        }
    }
    return SLATE_ERR_IO;
}
int slate_teacher_cache_size(slate_teacher_cache_t* c) { return c ? c->n_entries : 0; }
