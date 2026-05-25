// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#include "slate/sampling.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static float frand01(uint64_t* s) {
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((double)((*s >> 11) & ((1ULL << 53) - 1)) / (double)(1ULL << 53));
}

static int argmax(const float* p, int n) {
    int best = 0; float bv = p[0];
    for (int i = 1; i < n; ++i) if (p[i] > bv) { bv = p[i]; best = i; }
    return best;
}

static void top_k_indices(const float* p, int n, int k, int* idx) {
    for (int i = 0; i < k; ++i) idx[i] = i;
    int mp = 0;
    for (int i = 1; i < k; ++i) if (p[idx[i]] < p[idx[mp]]) mp = i;
    for (int i = k; i < n; ++i) {
        if (p[i] > p[idx[mp]]) {
            idx[mp] = i;
            mp = 0;
            for (int j = 1; j < k; ++j) if (p[idx[j]] < p[idx[mp]]) mp = j;
        }
    }
}

static int sample_from(const float* probs, const int* idx, int k, uint64_t* rng) {
    float r = frand01(rng);
    float cum = 0.0f;
    for (int i = 0; i < k; ++i) {
        cum += probs[i];
        if (r <= cum) return idx ? idx[i] : i;
    }
    return idx ? idx[k - 1] : k - 1;
}

int slate_sample_token(const float* logits, int V,
                        const slate_sampler_config_t* cfg, uint64_t* rng) {
    if (cfg->temperature <= 0.0f) return argmax(logits, V);
    float* probs = (float*)malloc((size_t)V * sizeof(float));
    float maxv = logits[0];
    for (int i = 1; i < V; ++i) if (logits[i] > maxv) maxv = logits[i];
    float inv_t = 1.0f / cfg->temperature;
    double sum = 0.0;
    for (int i = 0; i < V; ++i) { float e = expf((logits[i] - maxv) * inv_t); probs[i] = e; sum += e; }
    for (int i = 0; i < V; ++i) probs[i] = (float)(probs[i] / sum);

    if (cfg->top_k > 0 && cfg->top_k < V) {
        int* idx = (int*)malloc((size_t)cfg->top_k * sizeof(int));
        top_k_indices(probs, V, cfg->top_k, idx);
        float* tp = (float*)malloc((size_t)cfg->top_k * sizeof(float));
        double s = 0.0;
        for (int i = 0; i < cfg->top_k; ++i) s += probs[idx[i]];
        for (int i = 0; i < cfg->top_k; ++i) tp[i] = (float)(probs[idx[i]] / s);
        int tok;
        if (cfg->top_p > 0.0f && cfg->top_p < 1.0f) {
            // sort top-k descending
            for (int i = 1; i < cfg->top_k; ++i) {
                float pp = tp[i]; int ii = idx[i]; int j = i - 1;
                while (j >= 0 && tp[j] < pp) { tp[j+1]=tp[j]; idx[j+1]=idx[j]; --j; }
                tp[j+1] = pp; idx[j+1] = ii;
            }
            double cum = 0; int keep = 0;
            for (int i = 0; i < cfg->top_k; ++i) { cum += tp[i]; ++keep; if (cum >= cfg->top_p) break; }
            double ns = 0; for (int i = 0; i < keep; ++i) ns += tp[i];
            for (int i = 0; i < keep; ++i) tp[i] = (float)(tp[i] / ns);
            tok = sample_from(tp, idx, keep, rng);
        } else {
            tok = sample_from(tp, idx, cfg->top_k, rng);
        }
        free(tp); free(idx); free(probs);
        return tok;
    }
    int tok = sample_from(probs, NULL, V, rng);
    free(probs);
    return tok;
}
