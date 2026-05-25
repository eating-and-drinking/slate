// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#define _POSIX_C_SOURCE 200809L
#include "slate/mmap_dataset.h"
#include "slate/error.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct slate_mmap_dataset {
    int fd;
    const int32_t* tokens;
    int64_t n_tokens;
    size_t map_size;
};

slate_mmap_dataset_t* slate_mmap_open(const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    if (st.st_size < (off_t)sizeof(int32_t)) { close(fd); return NULL; }
    void* p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { close(fd); return NULL; }
    slate_mmap_dataset_t* d = (slate_mmap_dataset_t*)calloc(1, sizeof(*d));
    d->fd = fd;
    d->tokens = (const int32_t*)p;
    d->n_tokens = (int64_t)(st.st_size / sizeof(int32_t));
    d->map_size = (size_t)st.st_size;
    return d;
}

void slate_mmap_close(slate_mmap_dataset_t* d) {
    if (!d) return;
    munmap((void*)d->tokens, d->map_size);
    close(d->fd);
    free(d);
}

int64_t slate_mmap_n_tokens(const slate_mmap_dataset_t* d) { return d ? d->n_tokens : 0; }

int slate_mmap_sample_batch(slate_mmap_dataset_t* d, int B, int T,
                             int32_t* inputs, int32_t* targets,
                             uint64_t* rng) {
    if (!d || B <= 0 || T <= 0) return -1;
    if (d->n_tokens < (int64_t)(T + 1)) return -1;
    int64_t max_start = d->n_tokens - T - 1;
    for (int b = 0; b < B; ++b) {
        *rng = *rng * 6364136223846793005ULL + 1442695040888963407ULL;
        int64_t s = (int64_t)((*rng >> 11) % (uint64_t)(max_start + 1));
        for (int t = 0; t < T; ++t) {
            inputs[b * T + t] = d->tokens[s + t];
            targets[b * T + t] = d->tokens[s + t + 1];
        }
    }
    return 0;
}
