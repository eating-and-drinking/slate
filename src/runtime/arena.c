// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// arena.c — bump-pointer arena allocator.

#include "slate/runtime.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

struct slate_arena {
    uint8_t* base;
    size_t   capacity;
    size_t   used;
};

#define SLATE_DEFAULT_ALIGN 16

static size_t align_up(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

slate_arena_t* slate_arena_create(size_t capacity_bytes) {
    if (capacity_bytes == 0) return NULL;

    slate_arena_t* a = (slate_arena_t*)calloc(1, sizeof(*a));
    if (!a) return NULL;

    // Allocate page-aligned backing memory. We use plain malloc for portability;
    // a future improvement is to mmap so we can madvise(DONTNEED) on reset.
    a->base = (uint8_t*)malloc(capacity_bytes);
    if (!a->base) { free(a); return NULL; }

    a->capacity = capacity_bytes;
    a->used = 0;
    return a;
}

void slate_arena_destroy(slate_arena_t* a) {
    if (!a) return;
    free(a->base);
    free(a);
}

void* slate_arena_alloc(slate_arena_t* a, size_t bytes, size_t alignment) {
    if (!a || bytes == 0) return NULL;
    if (alignment == 0) alignment = SLATE_DEFAULT_ALIGN;
    // alignment must be a power of two.
    if ((alignment & (alignment - 1)) != 0) return NULL;

    size_t offset = align_up(a->used, alignment);
    if (offset > a->capacity || bytes > a->capacity - offset) {
        return NULL;  // would overflow
    }

    void* p = a->base + offset;
    a->used = offset + bytes;

    // Zero on allocate. This is the right default for tensor data and grad
    // buffers; callers that don't want it can overwrite immediately.
    memset(p, 0, bytes);
    return p;
}

size_t slate_arena_used(const slate_arena_t* a) {
    return a ? a->used : 0;
}

size_t slate_arena_capacity(const slate_arena_t* a) {
    return a ? a->capacity : 0;
}

void slate_arena_reset(slate_arena_t* a) {
    if (a) a->used = 0;
}
