// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// opd.c — On-Policy Distillation utilities. See include/slate/opd.h
// for the recipe; this file only implements the top-k extraction
// helper. The actual sample-then-distill training loop is host code
// (see examples/07_opd/main.c and tests/test_opd.c).

#include "slate/opd.h"

#include <stdint.h>

// One-position top-k via "smallest-of-running-K" min-heap-by-scan.
// Correct, O(V*K) — for K << V this is fast in cache, and it allocates
// nothing on the heap so it composes nicely with the arena world.
static void topk_one(const float* row, int V, int K,
                      int32_t* out_idx, float* out_val) {
    // Initialise with the first K entries.
    int K_eff = (K < V) ? K : V;
    for (int k = 0; k < K_eff; ++k) {
        out_idx[k] = (int32_t)k;
        out_val[k] = row[k];
    }
    // Find the index of the current minimum of out_val[0..K_eff).
    int min_pos = 0;
    for (int k = 1; k < K_eff; ++k)
        if (out_val[k] < out_val[min_pos]) min_pos = k;

    // Scan the rest, swapping in any value larger than the running min.
    for (int v = K_eff; v < V; ++v) {
        if (row[v] > out_val[min_pos]) {
            out_val[min_pos] = row[v];
            out_idx[min_pos] = (int32_t)v;
            // Recompute the running min position.
            min_pos = 0;
            for (int k = 1; k < K_eff; ++k)
                if (out_val[k] < out_val[min_pos]) min_pos = k;
        }
    }
    // If K > V (caller asked for more than vocab has), pad with -1 / -inf.
    for (int k = K_eff; k < K; ++k) {
        out_idx[k] = -1;
        out_val[k] = -1e30f;
    }
}

void slate_topk_extract(const float* logits,
                         int B, int T, int V, int K,
                         int32_t* out_indices,
                         float*   out_logits) {
    if (!logits || !out_indices || !out_logits) return;
    if (B <= 0 || T <= 0 || V <= 0 || K <= 0) return;
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const float* row = logits + ((int64_t)b * T + t) * V;
            int32_t* oi     = out_indices + ((int64_t)b * T + t) * K;
            float*   ov     = out_logits  + ((int64_t)b * T + t) * K;
            topk_one(row, V, K, oi, ov);
        }
    }
}
