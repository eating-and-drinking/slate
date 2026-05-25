// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// simple_dataloader.c — synchronous in-memory shuffler.

#include "slate/data_simple.h"

#include <stdlib.h>
#include <string.h>

// xoshiro256** PRNG, same as in linear.c. We duplicate the four lines because
// the runtime doesn't currently expose a global PRNG.
static uint64_t xrot(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
typedef struct { uint64_t s[4]; } xoshiro_t;
static void xoshiro_seed(xoshiro_t* r, uint64_t seed) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 4; ++i) {
        z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ULL;
        z ^= z >> 27; z *= 0x94D049BB133111EBULL;
        z ^= z >> 31;
        r->s[i] = z;
    }
}
static uint64_t xoshiro_next(xoshiro_t* r) {
    uint64_t result = xrot(r->s[1] * 5, 7) * 9;
    uint64_t t = r->s[1] << 17;
    r->s[2] ^= r->s[0];
    r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2];
    r->s[0] ^= r->s[3];
    r->s[2] ^= t;
    r->s[3] = xrot(r->s[3], 45);
    return result;
}

struct slate_simple_dataloader {
    int* perm;          // permutation buffer, length n
    int  n;
    int  batch;
    int  pos;           // next batch start index
    bool shuffle;
    xoshiro_t rng;
};

static void fisher_yates(int* a, int n, xoshiro_t* rng) {
    for (int i = n - 1; i > 0; --i) {
        int j = (int)(xoshiro_next(rng) % (uint64_t)(i + 1));
        int t = a[i]; a[i] = a[j]; a[j] = t;
    }
}

slate_simple_dataloader_t* slate_simple_dataloader_new(int n_samples,
                                                        int batch_size,
                                                        bool shuffle,
                                                        uint64_t seed) {
    if (n_samples <= 0 || batch_size <= 0) return NULL;
    slate_simple_dataloader_t* dl = (slate_simple_dataloader_t*)calloc(1, sizeof(*dl));
    if (!dl) return NULL;
    dl->n = n_samples;
    dl->batch = batch_size;
    dl->shuffle = shuffle;
    xoshiro_seed(&dl->rng, seed);
    dl->perm = (int*)malloc((size_t)n_samples * sizeof(int));
    if (!dl->perm) { free(dl); return NULL; }
    for (int i = 0; i < n_samples; ++i) dl->perm[i] = i;
    slate_simple_dataloader_reset(dl);
    return dl;
}

void slate_simple_dataloader_reset(slate_simple_dataloader_t* dl) {
    if (!dl) return;
    if (dl->shuffle) fisher_yates(dl->perm, dl->n, &dl->rng);
    dl->pos = 0;
}

bool slate_simple_dataloader_next(slate_simple_dataloader_t* dl, int* indices_out) {
    if (!dl || !indices_out) return false;
    if (dl->pos + dl->batch > dl->n) return false;  // drop_last
    memcpy(indices_out, dl->perm + dl->pos, (size_t)dl->batch * sizeof(int));
    dl->pos += dl->batch;
    return true;
}

int slate_simple_dataloader_n_batches(const slate_simple_dataloader_t* dl) {
    if (!dl) return 0;
    return dl->n / dl->batch;
}

void slate_simple_dataloader_destroy(slate_simple_dataloader_t* dl) {
    if (!dl) return;
    free(dl->perm);
    free(dl);
}
