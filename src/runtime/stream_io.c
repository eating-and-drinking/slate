// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#define _POSIX_C_SOURCE 200809L
#include "slate/stream_io.h"
#include "slate/error.h"
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define SLATE_SU_MAGIC "SLTSU\0\0"
#define SLATE_SU_HEADER 64

typedef struct {
    char     magic[8];
    uint32_t version;
    uint32_t dtype;
    uint32_t n_dims;
    uint32_t _pad;
    int64_t  shape[6];
    // total header is 64 bytes
} su_header_t;

slate_status_t slate_stream_write(const slate_tensor_t* t, const char* path) {
    if (!t || !path) return SLATE_ERR_INVALID_ARGUMENT;
    FILE* fp = fopen(path, "wb");
    if (!fp) return SLATE_ERR_IO;
    su_header_t h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, SLATE_SU_MAGIC, 8);
    h.version = 1;
    h.dtype = (uint32_t)t->dtype;
    h.n_dims = (uint32_t)t->n_dims;
    for (int i = 0; i < t->n_dims && i < 6; ++i) h.shape[i] = t->shape[i];
    if (fwrite(&h, 1, sizeof(h), fp) != sizeof(h)) { fclose(fp); return SLATE_ERR_IO; }
    size_t nb = slate_tensor_nbytes(t);
    if (fwrite(t->data, 1, nb, fp) != nb) { fclose(fp); return SLATE_ERR_IO; }
    fclose(fp);
    return SLATE_OK;
}

// We keep mmap bookkeeping in a small structure pointed to by t->grad_fn
// reinterpreted as opaque pointer. Cleaner: hijack t->grad as the saved
// mmap base + length so the user can later call slate_stream_release().
typedef struct stream_view {
    void* map_base;
    size_t map_len;
} stream_view_t;

slate_tensor_t* slate_stream_mmap(slate_arena_t* meta, const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st; if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    if (st.st_size < (off_t)sizeof(su_header_t)) { close(fd); return NULL; }
    void* p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { close(fd); return NULL; }
    close(fd);  // fd no longer needed after mmap

    su_header_t* h = (su_header_t*)p;
    if (memcmp(h->magic, SLATE_SU_MAGIC, 8) != 0) { munmap(p, (size_t)st.st_size); return NULL; }

    slate_tensor_t* t = (slate_tensor_t*)slate_arena_alloc(meta, sizeof(*t), 16);
    t->dtype = (slate_dtype_t)h->dtype;
    t->n_dims = (int)h->n_dims;
    int64_t stride_acc = (int64_t)slate_dtype_size(t->dtype);
    for (int i = (int)h->n_dims - 1; i >= 0; --i) {
        t->shape[i] = h->shape[i];
        t->stride[i] = stride_acc;
        stride_acc *= t->shape[i];
    }
    t->data = (uint8_t*)p + SLATE_SU_HEADER;
    t->grad = NULL;
    t->requires_grad = false;
    t->is_view = true;
    t->grad_fn = NULL;

    stream_view_t* sv = (stream_view_t*)slate_arena_alloc(meta, sizeof(*sv), 16);
    sv->map_base = p;
    sv->map_len = (size_t)st.st_size;
    // Stash sv into grad_fn (it's not used for non-grad views).
    t->grad_fn = (slate_graph_node_t*)sv;
    return t;
}

void slate_stream_release(slate_tensor_t* t) {
    if (!t || !t->grad_fn) return;
    stream_view_t* sv = (stream_view_t*)t->grad_fn;
    munmap(sv->map_base, sv->map_len);
    t->data = NULL;
    t->grad_fn = NULL;
}

size_t slate_stream_resident_bytes(const slate_tensor_t* t) {
    if (!t || !t->grad_fn) return 0;
    return ((const stream_view_t*)t->grad_fn)->map_len;
}
