// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// test_backend.c — verifies the slate_backend_t abstraction holds:
//   * slate_backend_cpu() returns a non-NULL vtable with all 13 entries
//   * slate_backend_default() falls back to cpu when CUDA/Metal stubs
//     are returning NULL
//   * Each compute primitive on the CPU backend produces results
//     bit-identical (or within fp32 epsilon) to direct reference
//     scalar implementations.
//
// This locks the abstraction so any future CUDA/Metal backend is held
// to the same input/output contract.

#include "slate/slate.h"
#include "slate/backend.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(void) {
    int ok = 1;
    const slate_backend_t* B = slate_backend_cpu();
    if (!B) { puts("backend_cpu = NULL FAIL"); return 1; }
    printf("backend.name = %s\n", B->name);

    // -- stubs return NULL --
    if (slate_backend_cuda()  != NULL) { puts("cuda stub != NULL FAIL");  ok = 0; }
    if (slate_backend_metal() != NULL) { puts("metal stub != NULL FAIL"); ok = 0; }
    if (slate_backend_default() != B) { puts("default != cpu FAIL"); ok = 0; }

    // ---- alloc/release + copy_h2d/d2h round-trip ----
    {
        const int N = 137;   // odd size to catch alignment bugs
        float host_src[137], host_dst[137];
        for (int i = 0; i < N; ++i) host_src[i] = (float)i * 0.5f - 12.0f;
        void* dev = B->alloc(N * sizeof(float));
        if (!dev) { puts("alloc FAIL"); return 1; }
        B->copy_h2d(dev, host_src, N * sizeof(float));
        B->copy_d2h(host_dst, dev, N * sizeof(float));
        float drift = max_abs_diff(host_src, host_dst, N);
        printf("[mem]      h2d -> d2h round-trip Linf = %.6e\n", drift);
        if (drift != 0) { puts("FAIL"); ok = 0; }
        B->release(dev);
    }

    // ---- matvec: y[M] = A[M, K] @ x[K] ----
    {
        const int M = 7, K = 11;
        float A[7 * 11], x[11], y[7], ref[7];
        for (int i = 0; i < M*K; ++i) A[i] = (float)((i * 13 + 5) % 17) * 0.05f - 0.4f;
        for (int k = 0; k < K; ++k) x[k] = (float)k * 0.1f - 0.5f;
        B->matvec(y, A, x, M, K);
        for (int m = 0; m < M; ++m) {
            double s = 0;
            for (int k = 0; k < K; ++k) s += (double)A[m*K + k] * (double)x[k];
            ref[m] = (float)s;
        }
        float drift = max_abs_diff(y, ref, M);
        printf("[matvec]   Linf vs reference = %.6e\n", drift);
        if (!(drift < 1e-4f)) { puts("FAIL"); ok = 0; }
    }

    // ---- linear_batch: C[B, N] = X[B, K] @ W[K, N] ----
    {
        const int BS = 4, K = 8, NN = 6;
        float X[4*8], W[8*6], C[4*6], ref[4*6];
        for (int i = 0; i < BS*K;  ++i) X[i] = (float)((i*7) % 13) * 0.1f - 0.6f;
        for (int i = 0; i < K*NN;  ++i) W[i] = (float)((i*5) % 11) * 0.08f - 0.4f;
        B->linear_batch(C, X, W, BS, K, NN);
        for (int b = 0; b < BS; ++b) for (int n = 0; n < NN; ++n) {
            double s = 0;
            for (int k = 0; k < K; ++k) s += (double)X[b*K + k] * (double)W[k*NN + n];
            ref[b*NN + n] = (float)s;
        }
        float drift = max_abs_diff(C, ref, BS*NN);
        printf("[linbatch] Linf vs reference = %.6e\n", drift);
        if (!(drift < 1e-4f)) { puts("FAIL"); ok = 0; }
    }

    // ---- rmsnorm_row ----
    {
        const int D = 16;
        float x[16], w[16], y[16], ref[16];
        for (int i = 0; i < D; ++i) {
            x[i] = (float)(i - 8) * 0.3f;
            w[i] = 0.5f + (float)i * 0.05f;
        }
        B->rmsnorm_row(y, x, w, D, 1e-5f);
        double sq = 0;
        for (int i = 0; i < D; ++i) sq += (double)x[i] * (double)x[i];
        float inv = 1.0f / sqrtf((float)(sq / D) + 1e-5f);
        for (int i = 0; i < D; ++i) ref[i] = x[i] * w[i] * inv;
        float drift = max_abs_diff(y, ref, D);
        printf("[rmsnorm]  Linf vs reference = %.6e\n", drift);
        if (!(drift < 1e-5f)) { puts("FAIL"); ok = 0; }
    }

    // ---- silu_mul ----
    {
        const int H = 24;
        float g[24], u[24], y[24], ref[24];
        for (int i = 0; i < H; ++i) {
            g[i] = (float)(i - 12) * 0.4f;
            u[i] = (float)(i + 1) * 0.1f;
            float s = g[i] / (1.0f + expf(-g[i]));
            ref[i] = s * u[i];
        }
        B->silu_mul(y, g, u, H);
        float drift = max_abs_diff(y, ref, H);
        printf("[silu_mul] Linf vs reference = %.6e\n", drift);
        if (!(drift < 1e-5f)) { puts("FAIL"); ok = 0; }
    }

    // ---- add_inplace ----
    {
        const int D = 32;
        float y[32], y0[32], x[32], ref[32];
        for (int i = 0; i < D; ++i) {
            y0[i] = (float)i * 0.1f;
            x[i]  = (float)(i - 16) * 0.2f;
            ref[i] = y0[i] + x[i];
        }
        memcpy(y, y0, sizeof(y));
        B->add_inplace(y, x, D);
        float drift = max_abs_diff(y, ref, D);
        printf("[add]      Linf vs reference = %.6e\n", drift);
        if (drift != 0) { puts("FAIL"); ok = 0; }
    }

    // ---- embed_lookup ----
    {
        const int V = 5, D = 8;
        float W[5 * 8], y[8], ref[8];
        for (int i = 0; i < V*D; ++i) W[i] = (float)i;
        int token_id = 3;
        B->embed_lookup(y, W, token_id, D);
        for (int i = 0; i < D; ++i) ref[i] = W[token_id*D + i];
        float drift = max_abs_diff(y, ref, D);
        printf("[embed]    Linf vs reference = %.6e\n", drift);
        if (drift != 0) { puts("FAIL"); ok = 0; }
    }

    // ---- attention_step ----
    {
        const int L = 5, D = 16;
        float q[16], K[5*16], V[5*16], scores[5], out[16], ref[16];
        for (int i = 0; i < D;   ++i) q[i] = (float)(i - 8) * 0.1f;
        for (int i = 0; i < L*D; ++i) K[i] = (float)((i*7) % 13) * 0.05f;
        for (int i = 0; i < L*D; ++i) V[i] = (float)((i*5) % 11) * 0.07f;
        B->attention_step(out, q, K, V, scores, L, D);
        // Reference: scores = q·K^T/sqrt(D); softmax; ·V
        double s_ref[5];
        double m = -1e30;
        for (int t = 0; t < L; ++t) {
            double s = 0;
            for (int d = 0; d < D; ++d) s += (double)q[d] * (double)K[t*D + d];
            s_ref[t] = s / sqrt((double)D);
            if (s_ref[t] > m) m = s_ref[t];
        }
        double S = 0;
        for (int t = 0; t < L; ++t) { s_ref[t] = exp(s_ref[t] - m); S += s_ref[t]; }
        for (int t = 0; t < L; ++t) s_ref[t] /= S;
        for (int d = 0; d < D; ++d) {
            double r = 0;
            for (int t = 0; t < L; ++t) r += s_ref[t] * (double)V[t*D + d];
            ref[d] = (float)r;
        }
        float drift = max_abs_diff(out, ref, D);
        printf("[attn]     Linf vs reference = %.6e\n", drift);
        if (!(drift < 1e-5f)) { puts("FAIL"); ok = 0; }
    }

    B->sync();  // should be a no-op
    printf("test_backend: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
