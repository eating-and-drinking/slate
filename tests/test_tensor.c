// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#include "slate/tensor.h"
#include "slate/runtime.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    slate_arena_t* arena = slate_arena_create(1024 * 1024);

    int64_t shape[2] = {3, 4};
    slate_tensor_t* t = slate_tensor_new(arena, SLATE_DTYPE_F32, 2, shape, true);
    assert(t);
    assert(t->n_dims == 2);
    assert(slate_tensor_numel(t) == 12);
    assert(slate_tensor_nbytes(t) == 48);
    assert(slate_tensor_is_contiguous(t));
    assert(t->grad != NULL);

    // Set / get round trip.
    float src[12];
    for (int i = 0; i < 12; ++i) src[i] = (float)i * 0.5f;
    assert(slate_tensor_set_data(t, src, sizeof(src)) == SLATE_OK);

    float dst[12];
    assert(slate_tensor_get_data(t, dst, sizeof(dst)) == SLATE_OK);
    for (int i = 0; i < 12; ++i) assert(dst[i] == src[i]);

    // Element accessor.
    float* p = slate_tensor_at_f32(t, 1, 2, 0, 0);
    assert(p != NULL);
    assert(*p == 6.0f * 0.5f);

    slate_arena_destroy(arena);
    printf("test_tensor: OK\n");
    return 0;
}
