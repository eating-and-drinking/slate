// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#include "slate/runtime.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    slate_arena_t* a = slate_arena_create(1024);
    assert(a);
    assert(slate_arena_capacity(a) == 1024);
    assert(slate_arena_used(a) == 0);

    void* p1 = slate_arena_alloc(a, 100, 16);
    assert(p1);
    assert(((uintptr_t)p1 & 15) == 0);
    assert(slate_arena_used(a) >= 100);

    void* p2 = slate_arena_alloc(a, 200, 32);
    assert(p2);
    assert(((uintptr_t)p2 & 31) == 0);
    assert(p2 != p1);

    // Allocation that would overflow returns NULL.
    void* p3 = slate_arena_alloc(a, 10000, 16);
    assert(!p3);

    // Reset re-enables allocation.
    slate_arena_reset(a);
    assert(slate_arena_used(a) == 0);
    void* p4 = slate_arena_alloc(a, 500, 16);
    assert(p4);

    // Zero-on-allocate.
    char* p5 = (char*)slate_arena_alloc(a, 100, 16);
    for (int i = 0; i < 100; ++i) assert(p5[i] == 0);

    slate_arena_destroy(a);
    printf("test_arena: OK\n");
    return 0;
}
