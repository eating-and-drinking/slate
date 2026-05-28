// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// llama_infer.c — LLaMA-architecture inference.  Reads weights from a
// slate_llama_t (which wraps a GGUF), runs multi-head causal
// self-attention with RoPE applied to Q and K per layer, SwiGLU FFN,
// tied (or untied) output projection.
//
// This file is the production-shape decode loop.  Compared to the
// existing src/infer/engine.c (which targets slate_module_causal_lm's
// single-head + additive pos-emb architecture), this one matches
// llama.cpp's behaviour:
//   * Multi-head: D = n_heads * head_dim, attention per head
//   * RoPE on Q and K right before the q·Kᵀ score matmul (post-RoPE
//     K is what gets cached, so it's already rotated for subsequent
//     decode steps)
//   * No pos_emb addition at the embedding layer
//   * Tied output: when slate_llama_t reports tied_output=1, the
//     same matrix serves as both the token embedding and the LM head
//
// MHA only for now (n_kv_heads == n_heads).  GQA is on the roadmap;
// the data layout already supports it (K and V have shape
// [d_model, n_kv_heads * head_dim]) but the attention loop hasn't
// been generalised yet.

#include "slate/llama.h"
#include "slate/backend.h"
#include "../ops/gemm_internal.h"
#include "../ops/simd_helpers.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// Dispatch to f32 matvec or Q4_K matvec based on the loaded weight dtype.
// SLATE_DTYPE_F32 = 0, SLATE_DTYPE_Q4_K = 18.
static inline void mv_dispatch(const slate_backend_t* B,
                                float* y, const void* W, int W_dtype,
                                const float* x, int M, int K) {
    if (W_dtype == 18 /* SLATE_DTYPE_Q4_K */ && B->matvec_q4k) {
        B->matvec_q4k(y, W, x, M, K);
    } else {
        B->matvec(y, (const float*)W, x, M, K);
    }
}

struct slate_llama_session {
    const slate_llama_t* model;
    const slate_backend_t* backend;
    int position;
    // KV caches: [n_layers, max_seq, n_kv_heads * head_dim]
    float* K_cache;
    float* V_cache;
    // Scratch (single-token wide)
    float* x;        // [D]
    float* x_norm;   // [D]
    float* q;        // [n_heads * head_dim]
    float* k;        // [n_kv_heads * head_dim]
    float* v;        // [n_kv_heads * head_dim]
    float* attn_out; // [n_heads * head_dim]
    float* ffn_g;    // [ffn_hidden]
    float* ffn_u;    // [ffn_hidden]
    float* scratch;  // [max(D, vocab)]
    float* scores;   // [max_seq]
};

// ---------------------------------------------------------------------------
// Per-head RoPE on a single-token Q (or K) row.
// q_row shape: [n_heads * head_dim].  For each head independently rotate.
// ---------------------------------------------------------------------------
static void rope_inplace_row(float* row, int n_heads, int head_dim,
                              int position, float theta_base) {
    int half = head_dim / 2;
    for (int h = 0; h < n_heads; ++h) {
        float* hh = row + h * head_dim;
        for (int i = 0; i < half; ++i) {
            float inv_freq = powf(theta_base, -2.0f * (float)i / (float)head_dim);
            float angle = (float)position * inv_freq;
            float c = cosf(angle), s = sinf(angle);
            float a = hh[i];
            float b = hh[i + half];
            hh[i]        = a * c - b * s;
            hh[i + half] = a * s + b * c;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-head causal attention against cached K and V.
// ---------------------------------------------------------------------------
static void attention_multihead(float* out,           // [n_q_heads * head_dim]
                                  const float* q,      // [n_q_heads * head_dim]
                                  const float* K_cache,// [L, n_kv_heads * head_dim]
                                  const float* V_cache,// [L, n_kv_heads * head_dim]
                                  float* scores,       // [L] scratch
                                  int L, int n_q_heads, int n_kv_heads, int head_dim) {
    int kv_dim = n_kv_heads * head_dim;
    int group  = n_q_heads / n_kv_heads;   // GQA group size (1 = MHA)
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int hq = 0; hq < n_q_heads; ++hq) {
        int hkv = hq / group;            // which K/V head this Q head reads
        const float* q_h = q + hq * head_dim;
        float* o_h = out + hq * head_dim;
        for (int t = 0; t < L; ++t) {
            const float* k_row = K_cache + (int64_t)t * kv_dim + hkv * head_dim;
            float s = 0;
            for (int d = 0; d < head_dim; ++d) s += q_h[d] * k_row[d];
            scores[t] = s * scale;
        }
        float m = scores[0];
        for (int t = 1; t < L; ++t) if (scores[t] > m) m = scores[t];
        double S = 0;
        for (int t = 0; t < L; ++t) { scores[t] = expf(scores[t] - m); S += scores[t]; }
        float invS = (float)(1.0 / S);
        for (int t = 0; t < L; ++t) scores[t] *= invS;
        for (int d = 0; d < head_dim; ++d) o_h[d] = 0;
        for (int t = 0; t < L; ++t) {
            const float* v_row = V_cache + (int64_t)t * kv_dim + hkv * head_dim;
            float w = scores[t];
            for (int d = 0; d < head_dim; ++d) o_h[d] += w * v_row[d];
        }
    }
}

// ---------------------------------------------------------------------------
// One decode step.  All weight pointers and shapes come from the model.
// ---------------------------------------------------------------------------
static int decode_one(slate_llama_session_t* sess, int32_t token, float* out_logits) {
    const slate_llama_t* m = sess->model;
    const slate_llama_config_t* c = slate_llama_config(m);
    int D = c->d_model;
    int H = c->ffn_hidden;
    int nh = c->n_heads;
    int hd = c->head_dim;
    int V = c->vocab;
    int max_seq = c->max_seq;
    int p = sess->position;
    if (p >= max_seq) return -2;
    if (token < 0 || token >= V) return -3;
    const slate_backend_t* B = sess->backend;

    // 1. Embed
    const float* tok_emb = (const float*)slate_llama_token_embd(m);
    memcpy(sess->x, tok_emb + (int64_t)token * D, (size_t)D * sizeof(float));

    // 2. Per-layer
    for (int li = 0; li < c->n_layers; ++li) {
        const slate_llama_layer_t* lw = slate_llama_layer(m, li);

        // a. Attention pre-norm
        B->rmsnorm_row(sess->x_norm, sess->x,
                        (const float*)lw->attn_norm, D, c->rms_eps);

        // b. Q, K, V projections
        mv_dispatch(B, sess->q, lw->attn_q, lw->attn_q_dtype, sess->x_norm, D, D);
        int kv_dim = c->n_kv_heads * hd;
        mv_dispatch(B, sess->k, lw->attn_k, lw->attn_k_dtype, sess->x_norm, kv_dim, D);
        mv_dispatch(B, sess->v, lw->attn_v, lw->attn_v_dtype, sess->x_norm, kv_dim, D);

        // c. RoPE on Q and K (per head, using current position)
        rope_inplace_row(sess->q, nh,           hd, p, c->theta_base);
        rope_inplace_row(sess->k, c->n_kv_heads, hd, p, c->theta_base);

        // d. Append K, V into cache
        float* K_l = sess->K_cache + (int64_t)li * max_seq * kv_dim;
        float* V_l = sess->V_cache + (int64_t)li * max_seq * kv_dim;
        memcpy(K_l + (int64_t)p * kv_dim, sess->k, (size_t)kv_dim * sizeof(float));
        memcpy(V_l + (int64_t)p * kv_dim, sess->v, (size_t)kv_dim * sizeof(float));

        // e. Multi-head causal attention (handles MHA and GQA — Q heads
        //    grouped to a smaller pool of K/V heads).
        attention_multihead(sess->attn_out, sess->q, K_l, V_l, sess->scores,
                             p + 1, nh, c->n_kv_heads, hd);

        // f. Output projection + residual
        mv_dispatch(B, sess->scratch, lw->attn_output, lw->attn_output_dtype, sess->attn_out, D, D);
        B->add_inplace(sess->x, sess->scratch, D);

        // g. FFN pre-norm
        B->rmsnorm_row(sess->x_norm, sess->x,
                        (const float*)lw->ffn_norm, D, c->rms_eps);

        // h. SwiGLU FFN
        mv_dispatch(B, sess->ffn_g, lw->ffn_gate, lw->ffn_gate_dtype, sess->x_norm, H, D);
        mv_dispatch(B, sess->ffn_u, lw->ffn_up, lw->ffn_up_dtype, sess->x_norm, H, D);
        B->silu_mul(sess->ffn_g, sess->ffn_g, sess->ffn_u, H);
        mv_dispatch(B, sess->scratch, lw->ffn_down, lw->ffn_down_dtype, sess->ffn_g, D, H);
        B->add_inplace(sess->x, sess->scratch, D);
    }

    // 3. Final norm
    B->rmsnorm_row(sess->x_norm, sess->x,
                    (const float*)slate_llama_output_norm(m), D, c->rms_eps);

    // 4. LM head
    mv_dispatch(B, out_logits, slate_llama_output(m), slate_llama_output_dtype(m), sess->x_norm, V, D);

    sess->position = p + 1;
    return 0;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

slate_llama_session_t* slate_llama_session_new(const slate_llama_t* model) {
    if (!model) return NULL;
    const slate_llama_config_t* c = slate_llama_config(model);
    slate_llama_session_t* s = (slate_llama_session_t*)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->model = model;
    s->backend = slate_backend_default();
    int D = c->d_model;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int H = c->ffn_hidden;
    int V = c->vocab;
    int max_seq = c->max_seq;
    int scratch_n = (D > V) ? D : V;
    s->K_cache = (float*)calloc((size_t)c->n_layers * max_seq * kv_dim, sizeof(float));
    s->V_cache = (float*)calloc((size_t)c->n_layers * max_seq * kv_dim, sizeof(float));
    s->x       = (float*)calloc((size_t)D, sizeof(float));
    s->x_norm  = (float*)calloc((size_t)D, sizeof(float));
    s->q       = (float*)calloc((size_t)D, sizeof(float));
    s->k       = (float*)calloc((size_t)kv_dim, sizeof(float));
    s->v       = (float*)calloc((size_t)kv_dim, sizeof(float));
    s->attn_out= (float*)calloc((size_t)D, sizeof(float));
    s->ffn_g   = (float*)calloc((size_t)H, sizeof(float));
    s->ffn_u   = (float*)calloc((size_t)H, sizeof(float));
    s->scratch = (float*)calloc((size_t)scratch_n, sizeof(float));
    s->scores  = (float*)calloc((size_t)max_seq, sizeof(float));
    if (!s->K_cache || !s->V_cache || !s->x || !s->x_norm || !s->q || !s->k ||
        !s->v || !s->attn_out || !s->ffn_g || !s->ffn_u || !s->scratch || !s->scores) {
        slate_llama_session_free(s);
        return NULL;
    }
    s->position = 0;
    return s;
}

void slate_llama_session_free(slate_llama_session_t* s) {
    if (!s) return;
    free(s->K_cache); free(s->V_cache);
    free(s->x); free(s->x_norm);
    free(s->q); free(s->k); free(s->v);
    free(s->attn_out);
    free(s->ffn_g); free(s->ffn_u);
    free(s->scratch);
    free(s->scores);
    free(s);
}

int slate_llama_session_position(const slate_llama_session_t* s) {
    return s ? s->position : -1;
}

int slate_llama_prefill(slate_llama_session_t* sess,
                         const int32_t* tokens, int n_tokens,
                         float* out_logits) {
    if (!sess || !tokens || !out_logits || n_tokens <= 0) return -1;
    int rc = 0;
    for (int i = 0; i < n_tokens; ++i) {
        rc = decode_one(sess, tokens[i], out_logits);
        if (rc != 0) return rc;
    }
    return 0;
}

int slate_llama_decode_step(slate_llama_session_t* sess,
                             int32_t token, float* out_logits) {
    if (!sess || !out_logits) return -1;
    return decode_one(sess, token, out_logits);
}
