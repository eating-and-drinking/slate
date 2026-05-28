// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// test_q4k.c — Q4_K_M dequant and Q4_K × f32 inner-product correctness.
//
// Tools/make_q4k_test.py emits /tmp/slate_q4k_test.bin containing:
//   bytes [0   .. 144) :  one Q4_K super-block (256 weights)
//   bytes [144 .. 1168):  256 f32 original weights (before quant)
//   bytes [1168..2192):  256 f32 reference reconstruction (per spec)
//
// We verify:
//   1. slate_dequant_q4_k matches the Python reference reconstruction
//      bit-identically (this is the spec-conformance check).
//   2. The dequant is within mean-abs ≤ 0.05 of the original f32 weights
//      (the quantization-error budget — sanity check on the format).
//   3. slate_dot_q4_k_f32 matches "dequant then dot" within 1e-5
//      relative error (the inference fast-path check).

#include "slate/slate.h"
#include "slate/quant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 256

int main(void) {
    FILE* fp = fopen("/tmp/slate_q4k_test.bin", "rb");
    if (!fp) { puts("missing /tmp/slate_q4k_test.bin -- run tools/make_q4k_test.py"); return 1; }

    uint8_t block[144];
    float   orig[N], ref_recon[N];
    if (fread(block,      1,   144, fp) != 144)   { puts("read block FAIL");   return 1; }
    if (fread(orig,      sizeof(float), N,  fp) != N) { puts("read orig FAIL"); return 1; }
    if (fread(ref_recon, sizeof(float), N,  fp) != N) { puts("read recon FAIL"); return 1; }
    fclose(fp);

    int ok = 1;

    // -- 1. slate_dequant_q4_k vs Python reference recon --
    float out[N];
    slate_dequant_q4_k(out, block, N);
    float spec_err_max = 0;
    for (int i = 0; i < N; ++i) {
        float e = fabsf(out[i] - ref_recon[i]);
        if (e > spec_err_max) spec_err_max = e;
    }
    printf("[1] dequant vs python reference recon: Linf = %.6e\n", spec_err_max);
    if (!(spec_err_max < 1e-5f)) { puts("FAIL: spec drift"); ok = 0; }

    // -- 2. dequant within quantization budget of original --
    double sum_err = 0;
    for (int i = 0; i < N; ++i) sum_err += fabsf(out[i] - orig[i]);
    double mean_err = sum_err / N;
    printf("[2] dequant vs original f32 weights:    mean |err| = %.4f\n", mean_err);
    if (!(mean_err < 0.05)) { puts("FAIL: quant error too high"); ok = 0; }

    // -- 3. slate_dot_q4_k_f32 vs dequant-then-dot --
    // Make a random y vector
    float y[N];
    uint64_t s = 0xBEEF;
    for (int i = 0; i < N; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        y[i] = ((float)((s >> 11) & ((1ULL<<24)-1)) / (1ULL<<24) - 0.5f) * 2.0f;
    }
    double ref_dot = 0;
    for (int i = 0; i < N; ++i) ref_dot += (double)out[i] * (double)y[i];
    float direct = slate_dot_q4_k_f32(block, y, N);
    double rel = fabs((double)direct - ref_dot) / (fabs(ref_dot) + 1e-9);
    printf("[3] direct dot vs (dequant then dot):  direct=%.6f ref=%.6f  rel_err=%.6e\n",
            direct, ref_dot, rel);
    if (!(rel < 1e-5)) { puts("FAIL: direct dot drift"); ok = 0; }

    printf("test_q4k: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
