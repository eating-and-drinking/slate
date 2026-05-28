// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// engine.c — CPU inference engine with KV cache.  See include/slate/infer.h
// for the public API and weight-layout convention.
//
// This file is independent from the training-side autograd code: it
// operates on raw float* buffers, calls the same packed-panel AVX2
// GEMM (slate_gemm_packed_accumulate) that powers training, and never
// touches the graph node arenas.  Per-token decode cost is O(L·D)
// instead of the prefill's O(L²·D), so generation latency stays
// roughly linear past a long prompt.

#include "slate/infer.h"
#include "slate/backend.h"
#include "slate/tensor.h"
#include "slate/module.h"
#include "../ops/gemm_internal.h"
#include "../ops/simd_helpers.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// ---------------------------------------------------------------------------
// Per-layer weight view
// ---------------------------------------------------------------------------

typedef struct {
    const float* norm1_w;  // [D]
    const float* Wq;       // [D, D]
    const float* Wk;       // [D, D]
    const float* Wv;       // [D, D]
    const float* Wo;       // [D, D]
    const float* norm2_w;  // [D]
    const float* Wg;       // [D, H]
    const float* Wu;       // [D, H]
    const float* Wd;       // [H, D]
} layer_w_t;

struct slate_infer_engine {
    int n_layers;
    int d_model;
    int vocab;
    int ffn_hidden;
    int max_seq;

    const float* tok_emb;     // [V, D]
    const float* pos_emb;     // [max_seq, D]
    const float* final_norm_w;// [D]
    const float* lm_head;     // [D, V]
    layer_w_t* layers;        // [n_layers]
    const slate_backend_t* backend;   // compute dispatch (CPU today; GPU plug-in)
};

struct slate_infer_session {
    slate_infer_engine_t* eng;
    int position;            // tokens already in cache (0..max_seq)
    // KV cache: per layer, contiguous [max_seq, D].
    float* K_cache;          // [n_layers * max_seq * D]
    float* V_cache;          // [n_layers * max_seq * D]
    // Scratch (single-token width).
    float* x;                // [D]
    float* x_norm;           // [D]
    float* q;                // [D]
    float* k;                // [D]
    float* v;                // [D]
    float* attn_out;         // [D]
    float* ffn_h_g;          // [H]
    float* ffn_h_u;          // [H]
    float* scores;           // [max_seq]
};

// ---------------------------------------------------------------------------
// Small primitives.  All single-token width unless noted.
// ---------------------------------------------------------------------------

// y[d] = x[d]   (single-row copy)
static inline void copy_d(float* y, const float* x, int D) {
    memcpy(y, x, (size_t)D * sizeof(float));
}

// y[d] += x[d]
static inline void add_d(float* y, const float* x, int D) {
#if defined(__AVX2__)
    int d = 0;
    for (; d + 8 <= D; d += 8) {
        __m256 a = _mm256_loadu_ps(y + d);
        __m256 b = _mm256_loadu_ps(x + d);
        _mm256_storeu_ps(y + d, _mm256_add_ps(a, b));
    }
    for (; d < D; ++d) y[d] += x[d];
#else
    for (int d = 0; d < D; ++d) y[d] += x[d];
#endif
}

// RMSNorm: y[d] = w[d] * x[d] / sqrt(mean(x^2) + eps)
static void rms_norm(float* y, const float* x, const float* w, int D, float eps) {
    // sum of squares
    double sq = 0;
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    int d0 = 0;
    for (; d0 + 8 <= D; d0 += 8) {
        __m256 xv = _mm256_loadu_ps(x + d0);
        acc = _mm256_fmadd_ps(xv, xv, acc);
    }
    sq = (double)slate_hsum256(acc);
    for (; d0 < D; ++d0) sq += (double)x[d0] * (double)x[d0];
#else
    for (int d = 0; d < D; ++d) sq += (double)x[d] * (double)x[d];
#endif
    float inv = 1.0f / sqrtf((float)(sq / (double)D) + eps);
    // y[d] = x[d] * w[d] * inv
#if defined(__AVX2__)
    __m256 iv = _mm256_set1_ps(inv);
    int d;
    for (d = 0; d + 8 <= D; d += 8) {
        __m256 xv = _mm256_loadu_ps(x + d);
        __m256 wv = _mm256_loadu_ps(w + d);
        _mm256_storeu_ps(y + d, _mm256_mul_ps(_mm256_mul_ps(xv, wv), iv));
    }
    for (; d < D; ++d) y[d] = x[d] * w[d] * inv;
#else
    for (int d = 0; d < D; ++d) y[d] = x[d] * w[d] * inv;
#endif
}

// y[N] = W[K, N]^T @ x[K]   ...   i.e. linear projection with row-major
// weight matrix shaped [in, out] (slate's convention).  out[j] = sum_k x[k]*W[k*N+j].
// Single-row version; uses the packed GEMM for the 1xK by KxN matmul.
static void linear_row(float* y, const float* x, const float* W, int K, int N) {
    // Zero output, then accumulate via GEMM with M=1.
    memset(y, 0, (size_t)N * sizeof(float));
    slate_gemm_packed_accumulate(y, /*ldc=*/N,
                                  x, /*lda=*/K,
                                  W, /*ldb=*/N,
                                  /*M=*/1, /*K=*/K, /*N=*/N);
}

// SiLU: y[d] = x[d] / (1 + exp(-x[d]))
static void silu_inplace(float* x, int H) {
#if defined(__AVX2__)
    int d = 0;
    for (; d + 8 <= H; d += 8) {
        __m256 v = _mm256_loadu_ps(x + d);
        __m256 s = slate_sigmoid256_ps(v);
        _mm256_storeu_ps(x + d, _mm256_mul_ps(v, s));
    }
    for (; d < H; ++d) {
        float v = x[d];
        x[d] = v / (1.0f + expf(-v));
    }
#else
    for (int d = 0; d < H; ++d) {
        float v = x[d];
        x[d] = v / (1.0f + expf(-v));
    }
#endif
}

// y[d] *= x[d]
static inline void mul_inplace(float* y, const float* x, int H) {
#if defined(__AVX2__)
    int d = 0;
    for (; d + 8 <= H; d += 8) {
        __m256 a = _mm256_loadu_ps(y + d);
        __m256 b = _mm256_loadu_ps(x + d);
        _mm256_storeu_ps(y + d, _mm256_mul_ps(a, b));
    }
    for (; d < H; ++d) y[d] *= x[d];
#else
    for (int d = 0; d < H; ++d) y[d] *= x[d];
#endif
}

// Single-token attention with causal cache.
//   q [D]          : current-position Q
//   K_cache[L, D]  : keys for positions 0..L-1
//   V_cache[L, D]  : values for positions 0..L-1
//   out [D]        : softmax(q @ K^T / sqrt(D)) @ V
//   scores [L]     : scratch
static void attention_step(float* out,
                            const float* q,
                            const float* K_cache,
                            const float* V_cache,
                            float* scores,
                            int L, int D) {
    float scale = 1.0f / sqrtf((float)D);
    // scores[t] = (q · K_cache[t, :]) * scale
#if defined(__AVX2__)
    for (int t = 0; t < L; ++t) {
        const float* k_row = K_cache + (int64_t)t * D;
        __m256 acc = _mm256_setzero_ps();
        int d = 0;
        for (; d + 8 <= D; d += 8) {
            __m256 qv = _mm256_loadu_ps(q + d);
            __m256 kv = _mm256_loadu_ps(k_row + d);
            acc = _mm256_fmadd_ps(qv, kv, acc);
        }
        float s = slate_hsum256(acc);
        for (; d < D; ++d) s += q[d] * k_row[d];
        scores[t] = s * scale;
    }
#else
    for (int t = 0; t < L; ++t) {
        const float* k_row = K_cache + (int64_t)t * D;
        float s = 0;
        for (int d = 0; d < D; ++d) s += q[d] * k_row[d];
        scores[t] = s * scale;
    }
#endif
    // softmax over scores[0..L)
    float m = scores[0];
    for (int t = 1; t < L; ++t) if (scores[t] > m) m = scores[t];
    double S = 0;
    for (int t = 0; t < L; ++t) { scores[t] = expf(scores[t] - m); S += scores[t]; }
    float invS = (float)(1.0 / S);
    for (int t = 0; t < L; ++t) scores[t] *= invS;
    // out[d] = sum_t scores[t] * V_cache[t, d]
    memset(out, 0, (size_t)D * sizeof(float));
    for (int t = 0; t < L; ++t) {
        const float* v_row = V_cache + (int64_t)t * D;
        float w = scores[t];
#if defined(__AVX2__)
        __m256 wv = _mm256_set1_ps(w);
        int d = 0;
        for (; d + 8 <= D; d += 8) {
            __m256 ov = _mm256_loadu_ps(out + d);
            __m256 vv = _mm256_loadu_ps(v_row + d);
            _mm256_storeu_ps(out + d, _mm256_fmadd_ps(wv, vv, ov));
        }
        for (; d < D; ++d) out[d] += w * v_row[d];
#else
        for (int d = 0; d < D; ++d) out[d] += w * v_row[d];
#endif
    }
}

// ---------------------------------------------------------------------------
// One inference step: token id at the current cache position -> logits.
// ---------------------------------------------------------------------------
static int decode_one(slate_infer_session_t* sess, int32_t token,
                       float* out_logits, float* extra_scratch) {
    slate_infer_engine_t* eng = sess->eng;
    const slate_backend_t* B = eng->backend;
    int D = eng->d_model;
    int H = eng->ffn_hidden;
    int L = eng->n_layers;
    int p = sess->position;
    if (p >= eng->max_seq) return -2;   // KV cache full
    if (token < 0 || token >= eng->vocab) return -3;

    // 1. Token embedding + position embedding -> x.
    B->embed_lookup(sess->x, eng->tok_emb, token, D);
    B->add_inplace(sess->x, eng->pos_emb + (int64_t)p * D, D);

    // 2. Per-layer transformer block (pre-norm).
    for (int li = 0; li < L; ++li) {
        const layer_w_t* lw = &eng->layers[li];
        // a. norm1
        B->rmsnorm_row(sess->x_norm, sess->x, lw->norm1_w, D, 1e-5f);
        // b. q, k, v
        B->linear_batch(sess->q, sess->x_norm, lw->Wq, 1, D, D);
        B->linear_batch(sess->k, sess->x_norm, lw->Wk, 1, D, D);
        B->linear_batch(sess->v, sess->x_norm, lw->Wv, 1, D, D);
        // c. append k, v into cache at position p
        float* K_cache_l = sess->K_cache + (int64_t)li * eng->max_seq * D;
        float* V_cache_l = sess->V_cache + (int64_t)li * eng->max_seq * D;
        memcpy(K_cache_l + (int64_t)p * D, sess->k, (size_t)D * sizeof(float));
        memcpy(V_cache_l + (int64_t)p * D, sess->v, (size_t)D * sizeof(float));
        // d. attention over positions 0..p (inclusive)
        B->attention_step(sess->attn_out, sess->q,
                        K_cache_l, V_cache_l, sess->scores,
                        /*L=*/p + 1, D);
        // e. output projection + residual
        B->linear_batch(extra_scratch, sess->attn_out, lw->Wo, 1, D, D);
        B->add_inplace(sess->x, extra_scratch, D);
        // f. norm2 + SwiGLU FFN
        B->rmsnorm_row(sess->x_norm, sess->x, lw->norm2_w, D, 1e-5f);
        B->linear_batch(sess->ffn_h_g, sess->x_norm, lw->Wg, 1, D, H);
        B->linear_batch(sess->ffn_h_u, sess->x_norm, lw->Wu, 1, D, H);
        B->silu_mul(sess->ffn_h_g, sess->ffn_h_g, sess->ffn_h_u, H);
        B->linear_batch(extra_scratch, sess->ffn_h_g, lw->Wd, 1, H, D);
        // g. residual
        B->add_inplace(sess->x, extra_scratch, D);
    }

    // 3. Final norm
    B->rmsnorm_row(sess->x_norm, sess->x, eng->final_norm_w, D, 1e-5f);

    // 4. LM head -> logits
    B->linear_batch(out_logits, sess->x_norm, eng->lm_head, 1, D, eng->vocab);

    sess->position = p + 1;
    return 0;
}

// ---------------------------------------------------------------------------
// Engine construction / destruction
// ---------------------------------------------------------------------------

slate_infer_engine_t* slate_infer_engine_new_ex(slate_module_t* model,
                                                 int n_layers,
                                                 int d_model,
                                                 int vocab,
                                                 int ffn_hidden,
                                                 int max_seq,
                                                 const slate_backend_t* backend) {
    if (!model || n_layers <= 0 || d_model <= 0 || vocab <= 0 ||
        ffn_hidden <= 0 || max_seq <= 0) return NULL;
    if (!backend) backend = slate_backend_default();
    if (!backend) return NULL;   // shouldn't happen — cpu is always available

    slate_param_set_t ps;
    slate_param_set_init(&ps);
    slate_module_register_params(model, &ps);
    int expected = 2 + n_layers * 9 + 2;
    if (ps.n_params != expected) {
        slate_param_set_destroy(&ps);
        return NULL;
    }

    slate_infer_engine_t* eng = (slate_infer_engine_t*)calloc(1, sizeof(*eng));
    if (!eng) { slate_param_set_destroy(&ps); return NULL; }
    eng->n_layers   = n_layers;
    eng->d_model    = d_model;
    eng->vocab      = vocab;
    eng->ffn_hidden = ffn_hidden;
    eng->max_seq    = max_seq;
    eng->backend    = backend;
    eng->tok_emb       = (const float*)ps.params[0]->data;
    eng->pos_emb       = (const float*)ps.params[1]->data;
    eng->layers = (layer_w_t*)calloc((size_t)n_layers, sizeof(layer_w_t));
    for (int i = 0; i < n_layers; ++i) {
        int b = 2 + i * 9;
        eng->layers[i].norm1_w = (const float*)ps.params[b + 0]->data;
        eng->layers[i].Wq      = (const float*)ps.params[b + 1]->data;
        eng->layers[i].Wk      = (const float*)ps.params[b + 2]->data;
        eng->layers[i].Wv      = (const float*)ps.params[b + 3]->data;
        eng->layers[i].Wo      = (const float*)ps.params[b + 4]->data;
        eng->layers[i].norm2_w = (const float*)ps.params[b + 5]->data;
        eng->layers[i].Wg      = (const float*)ps.params[b + 6]->data;
        eng->layers[i].Wu      = (const float*)ps.params[b + 7]->data;
        eng->layers[i].Wd      = (const float*)ps.params[b + 8]->data;
    }
    eng->final_norm_w = (const float*)ps.params[2 + n_layers*9 + 0]->data;
    eng->lm_head      = (const float*)ps.params[2 + n_layers*9 + 1]->data;
    slate_param_set_destroy(&ps);
    return eng;
}

// Legacy constructor — defaults to CPU backend.
slate_infer_engine_t* slate_infer_engine_new(slate_module_t* model,
                                              int n_layers,
                                              int d_model,
                                              int vocab,
                                              int ffn_hidden,
                                              int max_seq) {
    return slate_infer_engine_new_ex(model, n_layers, d_model, vocab,
                                      ffn_hidden, max_seq, NULL);
}

void slate_infer_engine_free(slate_infer_engine_t* eng) {
    if (!eng) return;
    free(eng->layers);
    free(eng);
}

int slate_infer_engine_vocab(const slate_infer_engine_t* eng) {
    return eng ? eng->vocab : 0;
}

int slate_infer_engine_max_seq(const slate_infer_engine_t* eng) {
    return eng ? eng->max_seq : 0;
}

slate_infer_session_t* slate_infer_session_new(slate_infer_engine_t* eng) {
    if (!eng) return NULL;
    slate_infer_session_t* s = (slate_infer_session_t*)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->eng = eng;
    int D = eng->d_model, H = eng->ffn_hidden;
    size_t cache_bytes = (size_t)eng->n_layers * eng->max_seq * D * sizeof(float);
    // KV cache + scratch all live on the backend's "device" — for CPU
    // backend that's host memory, for GPU it would be cudaMalloc.
    const slate_backend_t* B = eng->backend;
    s->K_cache   = (float*)B->alloc(cache_bytes);
    s->V_cache   = (float*)B->alloc(cache_bytes);
    s->x         = (float*)B->alloc((size_t)D * sizeof(float));
    s->x_norm    = (float*)B->alloc((size_t)D * sizeof(float));
    s->q         = (float*)B->alloc((size_t)D * sizeof(float));
    s->k         = (float*)B->alloc((size_t)D * sizeof(float));
    s->v         = (float*)B->alloc((size_t)D * sizeof(float));
    s->attn_out  = (float*)B->alloc((size_t)D * sizeof(float));
    s->ffn_h_g   = (float*)B->alloc((size_t)H * sizeof(float));
    s->ffn_h_u   = (float*)B->alloc((size_t)H * sizeof(float));
    s->scores    = (float*)B->alloc((size_t)eng->max_seq * sizeof(float));
    if (!s->K_cache || !s->V_cache || !s->x || !s->x_norm || !s->q ||
        !s->k || !s->v || !s->attn_out || !s->ffn_h_g || !s->ffn_h_u || !s->scores) {
        slate_infer_session_free(s);
        return NULL;
    }
    // Zero the KV cache; alloc returns uninitialised memory.
    memset(s->K_cache, 0, cache_bytes);
    memset(s->V_cache, 0, cache_bytes);
    s->position = 0;
    return s;
}

void slate_infer_session_free(slate_infer_session_t* sess) {
    if (!sess) return;
    const slate_backend_t* B = sess->eng->backend;
    B->release(sess->K_cache); B->release(sess->V_cache);
    B->release(sess->x);       B->release(sess->x_norm);
    B->release(sess->q);       B->release(sess->k);  B->release(sess->v);
    B->release(sess->attn_out);
    B->release(sess->ffn_h_g); B->release(sess->ffn_h_u);
    B->release(sess->scores);
    free(sess);
}

void slate_infer_session_reset(slate_infer_session_t* sess) {
    if (sess) sess->position = 0;
}

int slate_infer_session_position(const slate_infer_session_t* sess) {
    return sess ? sess->position : -1;
}

int slate_infer_prefill(slate_infer_session_t* sess,
                         const int32_t* tokens, int n_tokens,
                         float* out_logits) {
    if (!sess || !tokens || !out_logits || n_tokens <= 0) return -1;
    slate_infer_engine_t* eng = sess->eng;
    if (sess->position + n_tokens > eng->max_seq) return -2;
    // One linear_row(o, scratch, lm_head, D, V) scratch buffer; also Wo+Wd
    // residuals are written into a temporary, so we need a [max(D, V)] buffer.
    int max_dv = eng->vocab > eng->d_model ? eng->vocab : eng->d_model;
    float* extra = (float*)calloc((size_t)max_dv, sizeof(float));
    if (!extra) return -1;
    int rc = 0;
    for (int i = 0; i < n_tokens; ++i) {
        rc = decode_one(sess, tokens[i], out_logits, extra);
        if (rc != 0) break;
    }
    free(extra);
    return rc;
}

int slate_infer_decode_step(slate_infer_session_t* sess,
                             int32_t token, float* out_logits) {
    if (!sess || !out_logits) return -1;
    slate_infer_engine_t* eng = sess->eng;
    int max_dv = eng->vocab > eng->d_model ? eng->vocab : eng->d_model;
    float* extra = (float*)calloc((size_t)max_dv, sizeof(float));
    if (!extra) return -1;
    int rc = decode_one(sess, token, out_logits, extra);
    free(extra);
    return rc;
}

// ---------------------------------------------------------------------------
// Batched inference: continuous batching for B concurrent sessions.
// ---------------------------------------------------------------------------
//
// The hot loop is the same as decode_one but with all the linear
// projections run as M=B GEMMs rather than B separate M=1 GEMMs. The
// attention step stays per-session because each session has its own
// KV cache length.

struct slate_infer_batch {
    slate_infer_engine_t* eng;
    int max_batch;
    float* X;        // [B, D]
    float* X_norm;   // [B, D]
    float* Q;        // [B, D]
    float* Kp;       // [B, D]
    float* Vp;       // [B, D]
    float* attn_out; // [B, D]
    float* FFN_g;    // [B, H]
    float* FFN_u;    // [B, H]
    float* extra;    // [B, max(D, V)]
};

// C [M, N] = X [M, K] @ W [K, N]   via packed GEMM (single call).
static void linear_batch(float* C, const float* X, const float* W,
                          int M, int K, int N) {
    memset(C, 0, (size_t)M * (size_t)N * sizeof(float));
    slate_gemm_packed_accumulate(C, /*ldc=*/N,
                                  X, /*lda=*/K,
                                  W, /*ldb=*/N,
                                  M, K, N);
}

slate_infer_batch_t* slate_infer_batch_new(slate_infer_engine_t* eng,
                                            int max_batch) {
    if (!eng || max_batch <= 0) return NULL;
    slate_infer_batch_t* b = (slate_infer_batch_t*)calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->eng = eng;
    b->max_batch = max_batch;
    int D = eng->d_model;
    int H = eng->ffn_hidden;
    int V = eng->vocab;
    int extra_n = (D > V ? D : V);
    size_t B = (size_t)max_batch;
    const slate_backend_t* bk = eng->backend;
    b->X        = (float*)bk->alloc(B * D * sizeof(float));
    b->X_norm   = (float*)bk->alloc(B * D * sizeof(float));
    b->Q        = (float*)bk->alloc(B * D * sizeof(float));
    b->Kp       = (float*)bk->alloc(B * D * sizeof(float));
    b->Vp       = (float*)bk->alloc(B * D * sizeof(float));
    b->attn_out = (float*)bk->alloc(B * D * sizeof(float));
    b->FFN_g    = (float*)bk->alloc(B * H * sizeof(float));
    b->FFN_u    = (float*)bk->alloc(B * H * sizeof(float));
    b->extra    = (float*)bk->alloc(B * (size_t)extra_n * sizeof(float));
    if (!b->X || !b->X_norm || !b->Q || !b->Kp || !b->Vp ||
        !b->attn_out || !b->FFN_g || !b->FFN_u || !b->extra) {
        slate_infer_batch_free(b);
        return NULL;
    }
    return b;
}

void slate_infer_batch_free(slate_infer_batch_t* b) {
    if (!b) return;
    const slate_backend_t* bk = b->eng->backend;
    bk->release(b->X); bk->release(b->X_norm);
    bk->release(b->Q); bk->release(b->Kp); bk->release(b->Vp);
    bk->release(b->attn_out);
    bk->release(b->FFN_g); bk->release(b->FFN_u);
    bk->release(b->extra);
    free(b);
}

int slate_infer_batch_step(slate_infer_batch_t* batch,
                            slate_infer_session_t** sessions,
                            int n_sessions,
                            const int32_t* tokens,
                            float* out_logits) {
    if (!batch || !sessions || !tokens || !out_logits) return -1;
    if (n_sessions < 1 || n_sessions > batch->max_batch) return -1;
    slate_infer_engine_t* eng = batch->eng;
    int D = eng->d_model, H = eng->ffn_hidden, V = eng->vocab;
    int L = eng->n_layers;
    const slate_backend_t* bk = eng->backend;

    // 1. Build batched X [B, D]: tok_emb[token_r] + pos_emb[pos_r]
    for (int r = 0; r < n_sessions; ++r) {
        if (sessions[r]->eng != eng) return -1;
        if (sessions[r]->position >= eng->max_seq) return -2;
        if (tokens[r] < 0 || tokens[r] >= V) return -3;
        float* x_r = batch->X + (int64_t)r * D;
        bk->embed_lookup(x_r, eng->tok_emb, tokens[r], D);
        bk->add_inplace(x_r, eng->pos_emb + (int64_t)sessions[r]->position * D, D);
    }

    // 2. Per-layer transformer block
    for (int li = 0; li < L; ++li) {
        const layer_w_t* lw = &eng->layers[li];

        // a. norm1 per row
        for (int r = 0; r < n_sessions; ++r) {
            bk->rmsnorm_row(batch->X_norm + (int64_t)r * D,
                              batch->X      + (int64_t)r * D,
                              lw->norm1_w, D, 1e-5f);
        }

        // b. batched Q, K, V projections
        bk->linear_batch(batch->Q,  batch->X_norm, lw->Wq, n_sessions, D, D);
        bk->linear_batch(batch->Kp, batch->X_norm, lw->Wk, n_sessions, D, D);
        bk->linear_batch(batch->Vp, batch->X_norm, lw->Wv, n_sessions, D, D);

        // c. per-session attention against own cache
        for (int r = 0; r < n_sessions; ++r) {
            int p = sessions[r]->position;
            float* K_cache_l = sessions[r]->K_cache + (int64_t)li * eng->max_seq * D;
            float* V_cache_l = sessions[r]->V_cache + (int64_t)li * eng->max_seq * D;
            memcpy(K_cache_l + (int64_t)p * D,
                    batch->Kp + (int64_t)r * D,
                    (size_t)D * sizeof(float));
            memcpy(V_cache_l + (int64_t)p * D,
                    batch->Vp + (int64_t)r * D,
                    (size_t)D * sizeof(float));
            bk->attention_step(batch->attn_out + (int64_t)r * D,
                                batch->Q          + (int64_t)r * D,
                                K_cache_l, V_cache_l, sessions[r]->scores,
                                p + 1, D);
        }

        // d. Wo + residual
        bk->linear_batch(batch->extra, batch->attn_out, lw->Wo, n_sessions, D, D);
        for (int r = 0; r < n_sessions; ++r) {
            bk->add_inplace(batch->X + (int64_t)r * D,
                              batch->extra + (int64_t)r * D, D);
        }

        // e. norm2
        for (int r = 0; r < n_sessions; ++r) {
            bk->rmsnorm_row(batch->X_norm + (int64_t)r * D,
                              batch->X      + (int64_t)r * D,
                              lw->norm2_w, D, 1e-5f);
        }

        // f. SwiGLU FFN, batched
        bk->linear_batch(batch->FFN_g, batch->X_norm, lw->Wg, n_sessions, D, H);
        bk->linear_batch(batch->FFN_u, batch->X_norm, lw->Wu, n_sessions, D, H);
        for (int r = 0; r < n_sessions; ++r) {
            bk->silu_mul(batch->FFN_g + (int64_t)r * H,
                          batch->FFN_g + (int64_t)r * H,
                          batch->FFN_u + (int64_t)r * H, H);
        }
        bk->linear_batch(batch->extra, batch->FFN_g, lw->Wd, n_sessions, H, D);
        for (int r = 0; r < n_sessions; ++r) {
            bk->add_inplace(batch->X + (int64_t)r * D,
                              batch->extra + (int64_t)r * D, D);
        }
    }

    // 3. Final norm (per row)
    for (int r = 0; r < n_sessions; ++r) {
        bk->rmsnorm_row(batch->X_norm + (int64_t)r * D,
                          batch->X      + (int64_t)r * D,
                          eng->final_norm_w, D, 1e-5f);
    }

    // 4. LM head: out_logits [B, V] = X_norm [B, D] @ lm_head [D, V]
    bk->linear_batch(out_logits, batch->X_norm, eng->lm_head, n_sessions, D, V);

    // 5. Advance each session's position
    for (int r = 0; r < n_sessions; ++r) {
        sessions[r]->position++;
    }
    return 0;
}
