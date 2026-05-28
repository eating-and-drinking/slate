// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// test_gguf_q4k.c — end-to-end: GGUF file with Q4_K_M tensor → mmap →
// slate_dequant_q4_k → compare against the reference reconstruction emitted
// by tools/make_q4k_gguf.py.  Also exercises slate_dot_q4_k_f32 against a
// random y vector with double-precision reference dot.

#include "slate/slate.h"
#include "slate/gguf.h"
#include "slate/quant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_ROWS  16
#define QK_K   256
#define N      (N_ROWS * QK_K)

int main(void) {
    slate_gguf_t* g = slate_gguf_open("/tmp/slate_q4k.gguf");
    if (!g) { puts("gguf open FAIL (run tools/make_q4k_gguf.py)"); return 1; }
    printf("gguf opened, n_tensors=%d\n", slate_gguf_n_tensors(g));

    slate_arena_t* meta = slate_arena_create(64 * 1024);
    slate_tensor_t* W = slate_gguf_get_tensor(meta, g, "weight");
    if (!W) { puts("tensor 'weight' not found"); return 1; }
    printf("W shape=[%lld, %lld] dtype=%d\n",
            (long long)W->shape[0], (long long)W->shape[1], (int)W->dtype);
    int ok = (W->dtype == SLATE_DTYPE_Q4_K);
    if (!ok) { puts("FAIL: wrong dtype (expected Q4_K)"); }

    // Load expected reconstruction
    float expected[N];
    FILE* ef = fopen("/tmp/slate_q4k_expected.f32", "rb");
    if (!ef) { puts("expected file FAIL"); return 1; }
    if (fread(expected, sizeof(float), N, ef) != N) { puts("expected read FAIL"); return 1; }
    fclose(ef);

    // Dequant the entire tensor
    float dequant[N];
    slate_dequant_q4_k(dequant, W->data, N);

    float linf = 0;
    for (int i = 0; i < N; ++i) {
        float d = fabsf(dequant[i] - expected[i]);
        if (d > linf) linf = d;
    }
    printf("[dequant] Linf vs reference recon = %.6e   (must be 0)\n", linf);
    if (linf != 0) { puts("FAIL: dequant drifted from reference"); ok = 0; }

    // Random y vector + direct-dot vs dequant-then-dot
    float y[N];
    uint64_t s = 0xC0FFEE;
    for (int i = 0; i < N; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        y[i] = ((float)((s >> 24) & 0xFFFF) / 65536.0f - 0.5f);
    }
    double ref_dot = 0;
    for (int i = 0; i < N; ++i) ref_dot += (double)dequant[i] * (double)y[i];
    float direct = slate_dot_q4_k_f32(W->data, y, N);
    double rel = fabs((double)direct - ref_dot) / (fabs(ref_dot) + 1e-9);
    printf("[dot]     direct=%.6f  ref=%.6f  rel_err=%.6e\n",
            direct, ref_dot, rel);
    if (!(rel < 1e-5)) { puts("FAIL: direct dot drifted"); ok = 0; }

    // matvec end-to-end: y' = A @ x where A is the Q4_K matrix
    float x[QK_K];
    s = 0xABCD;
    for (int i = 0; i < QK_K; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        x[i] = ((float)((s >> 24) & 0xFFFF) / 65536.0f - 0.5f);
    }
    float y_out[N_ROWS];
    slate_q4_k_matvec(y_out, W->data, x, N_ROWS, QK_K);
    // Reference using dequant
    float y_ref[N_ROWS];
    for (int m = 0; m < N_ROWS; ++m) {
        double dot = 0;
        for (int k = 0; k < QK_K; ++k) dot += (double)dequant[m * QK_K + k] * x[k];
        y_ref[m] = (float)dot;
    }
    float matvec_linf = 0;
    for (int m = 0; m < N_ROWS; ++m) {
        float d = fabsf(y_out[m] - y_ref[m]) / (fabsf(y_ref[m]) + 1e-9f);
        if (d > matvec_linf) matvec_linf = d;
    }
    printf("[matvec]  q4_k_matvec vs dequant-and-dot rel Linf = %.6e\n", matvec_linf);
    if (!(matvec_linf < 1e-4f)) { puts("FAIL: matvec drift"); ok = 0; }

    slate_gguf_close(g);
    slate_arena_destroy(meta);
    printf("test_gguf_q4k: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
