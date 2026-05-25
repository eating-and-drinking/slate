// SPDX-License-Identifier: Apache-2.0
#include "slate/slate.h"
#include "slate/gguf.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    slate_gguf_t* g = slate_gguf_open("/tmp/slate_test.gguf");
    if (!g) { puts("gguf open FAIL"); return 1; }
    slate_gguf_dump(g);
    int n = slate_gguf_n_tensors(g);
    int ok = (n == 2);
    printf("[gguf] n_tensors=%d (expected 2)\n", n);

    slate_arena_t* meta = slate_arena_create(64 * 1024);
    slate_tensor_t* W = slate_gguf_get_tensor(meta, g, "test.weight_a");
    if (!W) { puts("[gguf] tensor lookup FAIL"); return 1; }
    printf("[gguf] W shape=[%lld, %lld]\n", (long long)W->shape[0], (long long)W->shape[1]);
    ok = ok && (W->n_dims == 2 && W->shape[0] == 4 && W->shape[1] == 3);
    float* p = (float*)W->data;
    printf("[gguf] W values: ");
    for (int i = 0; i < 12; ++i) printf("%.2f ", p[i]);
    printf("\n");
    // Expected: 0.1, 0.2, ..., 1.2
    int values_ok = 1;
    for (int i = 0; i < 12; ++i) {
        float expected = 0.1f * (i + 1);
        if (fabsf(p[i] - expected) > 1e-5f) { values_ok = 0; break; }
    }
    ok = ok && values_ok;
    printf("[gguf] W values match: %s\n", values_ok ? "yes" : "NO");

    slate_tensor_t* B = slate_gguf_get_tensor(meta, g, "test.bias_b");
    float* pb = (float*)B->data;
    printf("[gguf] B values: %.2f %.2f %.2f (expected 10,20,30)\n", pb[0], pb[1], pb[2]);
    int b_ok = (pb[0] == 10.0f && pb[1] == 20.0f && pb[2] == 30.0f);
    ok = ok && b_ok;

    slate_arena_destroy(meta);
    slate_gguf_close(g);
    printf("test_gguf: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
