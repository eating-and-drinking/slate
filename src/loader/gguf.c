// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#define _POSIX_C_SOURCE 200809L
#include "slate/gguf.h"
#include "slate/error.h"
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGUF_MAGIC 0x46554747u
#define GGUF_DEFAULT_ALIGN 32

enum {
    GGUF_T_U8=0, GGUF_T_I8, GGUF_T_U16, GGUF_T_I16,
    GGUF_T_U32, GGUF_T_I32, GGUF_T_F32, GGUF_T_BOOL,
    GGUF_T_STRING=8, GGUF_T_ARRAY=9, GGUF_T_U64=10, GGUF_T_I64=11, GGUF_T_F64=12,
};
enum {
    GGML_T_F32=0, GGML_T_F16=1,
    GGML_T_Q4_0=2, GGML_T_Q4_1=3, GGML_T_Q5_0=6, GGML_T_Q5_1=7,
    GGML_T_Q8_0=8, GGML_T_Q8_1=9,
};

typedef struct {
    char* name;
    int n_dims;
    int64_t shape[SLATE_MAX_DIMS];
    int dtype;
    uint64_t offset;
    size_t nbytes;
} gguf_tinfo_t;

struct slate_gguf {
    int fd;
    void* map;
    size_t map_size;
    uint64_t n_tensors;
    uint64_t n_kv;
    uint64_t alignment;
    gguf_tinfo_t* tinfos;
    uint64_t data_offset;
};

typedef struct { const uint8_t* p; const uint8_t* end; } cur_t;
static int cur_read(cur_t* c, void* dst, size_t n) {
    if (c->p + n > c->end) return -1;
    memcpy(dst, c->p, n); c->p += n; return 0;
}
static int cur_skip(cur_t* c, size_t n) {
    if (c->p + n > c->end) return -1; c->p += n; return 0;
}
static int cur_read_str(cur_t* c, char** out) {
    uint64_t len;
    if (cur_read(c, &len, 8)) return -1;
    if (len > 1024 * 1024 || c->p + len > c->end) return -1;
    char* s = (char*)malloc((size_t)len + 1);
    memcpy(s, c->p, (size_t)len); s[len] = 0; c->p += len; *out = s; return 0;
}
static size_t value_size(uint32_t t) {
    switch (t) {
        case GGUF_T_U8: case GGUF_T_I8: case GGUF_T_BOOL: return 1;
        case GGUF_T_U16: case GGUF_T_I16: return 2;
        case GGUF_T_U32: case GGUF_T_I32: case GGUF_T_F32: return 4;
        case GGUF_T_U64: case GGUF_T_I64: case GGUF_T_F64: return 8;
        default: return 0;
    }
}
static int skip_value(cur_t* c, uint32_t type) {
    if (type == GGUF_T_STRING) {
        char* s; if (cur_read_str(c, &s)) return -1; free(s); return 0;
    }
    if (type == GGUF_T_ARRAY) {
        uint32_t et; uint64_t n;
        if (cur_read(c, &et, 4)) return -1;
        if (cur_read(c, &n, 8)) return -1;
        for (uint64_t i = 0; i < n; ++i) if (skip_value(c, et)) return -1;
        return 0;
    }
    size_t sz = value_size(type);
    if (sz == 0) return -1;
    return cur_skip(c, sz);
}
static size_t dtype_nbytes(int t, int64_t numel) {
    if (t == GGML_T_F32) return (size_t)numel * 4;
    if (t == GGML_T_F16) return (size_t)numel * 2;
    if (t == GGML_T_Q8_0) return ((size_t)numel / 32) * 34;
    if (t == GGML_T_Q4_0) return ((size_t)numel / 32) * 18;
    return 0;
}
static slate_dtype_t map_dtype(int g) {
    switch (g) {
        case GGML_T_F32:  return SLATE_DTYPE_F32;
        case GGML_T_F16:  return SLATE_DTYPE_F16;
        case GGML_T_Q8_0: return SLATE_DTYPE_Q8_0;
        case GGML_T_Q4_0: return SLATE_DTYPE_Q4_0;
        default: return SLATE_DTYPE_F32;
    }
}

slate_gguf_t* slate_gguf_open(const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st; if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    void* p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { close(fd); return NULL; }
    cur_t c = { (const uint8_t*)p, (const uint8_t*)p + st.st_size };
    uint32_t magic, version;
    if (cur_read(&c, &magic, 4) || magic != GGUF_MAGIC) goto bad;
    if (cur_read(&c, &version, 4) || version < 2 || version > 3) goto bad;
    uint64_t n_tensors, n_kv;
    if (cur_read(&c, &n_tensors, 8)) goto bad;
    if (cur_read(&c, &n_kv, 8)) goto bad;
    uint64_t alignment = GGUF_DEFAULT_ALIGN;
    for (uint64_t i = 0; i < n_kv; ++i) {
        char* k; if (cur_read_str(&c, &k)) goto bad;
        uint32_t vt; if (cur_read(&c, &vt, 4)) { free(k); goto bad; }
        if (strcmp(k, "general.alignment") == 0 && vt == GGUF_T_U32) {
            uint32_t a; cur_read(&c, &a, 4); alignment = a;
        } else if (skip_value(&c, vt)) { free(k); goto bad; }
        free(k);
    }
    gguf_tinfo_t* tinfos = (gguf_tinfo_t*)calloc((size_t)n_tensors, sizeof(*tinfos));
    for (uint64_t i = 0; i < n_tensors; ++i) {
        if (cur_read_str(&c, &tinfos[i].name)) goto bad;
        uint32_t nd; if (cur_read(&c, &nd, 4)) goto bad;
        if (nd > SLATE_MAX_DIMS) goto bad;
        tinfos[i].n_dims = (int)nd;
        for (uint32_t d = 0; d < nd; ++d) {
            uint64_t s; if (cur_read(&c, &s, 8)) goto bad;
            tinfos[i].shape[d] = (int64_t)s;
        }
        uint32_t dt; if (cur_read(&c, &dt, 4)) goto bad;
        tinfos[i].dtype = (int)dt;
        if (cur_read(&c, &tinfos[i].offset, 8)) goto bad;
        int64_t numel = 1;
        for (int d = 0; d < tinfos[i].n_dims; ++d) numel *= tinfos[i].shape[d];
        tinfos[i].nbytes = dtype_nbytes(tinfos[i].dtype, numel);
    }
    size_t cur_off = (size_t)(c.p - (const uint8_t*)p);
    size_t pad = (alignment - (cur_off % alignment)) % alignment;
    size_t data_off = cur_off + pad;
    slate_gguf_t* g = (slate_gguf_t*)calloc(1, sizeof(*g));
    g->fd = fd; g->map = p; g->map_size = (size_t)st.st_size;
    g->n_tensors = n_tensors; g->n_kv = n_kv;
    g->alignment = alignment; g->tinfos = tinfos;
    g->data_offset = data_off;
    return g;
bad:
    munmap(p, (size_t)st.st_size); close(fd); return NULL;
}

void slate_gguf_close(slate_gguf_t* g) {
    if (!g) return;
    for (uint64_t i = 0; i < g->n_tensors; ++i) free(g->tinfos[i].name);
    free(g->tinfos);
    munmap(g->map, g->map_size); close(g->fd); free(g);
}
int slate_gguf_n_tensors(const slate_gguf_t* g) { return g ? (int)g->n_tensors : 0; }
const char* slate_gguf_tensor_name(const slate_gguf_t* g, int idx) {
    if (!g || idx < 0 || (uint64_t)idx >= g->n_tensors) return NULL;
    return g->tinfos[idx].name;
}
slate_tensor_t* slate_gguf_get_tensor(slate_arena_t* meta, slate_gguf_t* g, const char* name) {
    if (!g || !name) return NULL;
    gguf_tinfo_t* ti = NULL;
    for (uint64_t i = 0; i < g->n_tensors; ++i)
        if (strcmp(g->tinfos[i].name, name) == 0) { ti = &g->tinfos[i]; break; }
    if (!ti) return NULL;
    slate_tensor_t* t = (slate_tensor_t*)slate_arena_alloc(meta, sizeof(*t), 16);
    t->dtype = map_dtype(ti->dtype);
    t->n_dims = ti->n_dims;
    int64_t stride_acc = (int64_t)slate_dtype_size(t->dtype);
    for (int d = ti->n_dims - 1; d >= 0; --d) {
        t->shape[d] = ti->shape[d];
        t->stride[d] = stride_acc;
        stride_acc *= t->shape[d];
    }
    t->data = (uint8_t*)g->map + g->data_offset + ti->offset;
    t->grad = NULL; t->requires_grad = false; t->is_view = true; t->grad_fn = NULL;
    return t;
}
void slate_gguf_dump(const slate_gguf_t* g) {
    if (!g) return;
    printf("GGUF: %d tensors, %d kv, alignment=%d\n",
           (int)g->n_tensors, (int)g->n_kv, (int)g->alignment);
    for (uint64_t i = 0; i < g->n_tensors && i < 20; ++i) {
        gguf_tinfo_t* t = &g->tinfos[i];
        printf("  [%2d] dt=%d shape=[", (int)i, t->dtype);
        for (int d = 0; d < t->n_dims; ++d) printf("%lld%s", (long long)t->shape[d], d+1<t->n_dims?",":"");
        printf("] off=%llu nbytes=%zu  %s\n",
               (unsigned long long)t->offset, t->nbytes, t->name);
    }
    if (g->n_tensors > 20) printf("  ... (+%d)\n", (int)g->n_tensors - 20);
}
