// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// llama.c — LLaMA-format GGUF wrapper.  Parses llama.cpp's tensor
// naming convention and `llama.*` metadata keys into a structured
// view that the inference engine can consume.

#include "slate/llama.h"
#include "slate/gguf.h"
#include "slate/runtime.h"
#include "slate/tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct slate_llama {
    slate_gguf_t* gguf;
    slate_llama_config_t cfg;
    slate_arena_t* meta;            // arena owning the slate_tensor_t views

    const void* token_embd;
    int         token_embd_dtype;
    const void* output_norm;
    const void* output;
    int         output_dtype;

    slate_llama_layer_t* layers;    // [n_layers]
};

// Helper: resolve a named tensor and return its data pointer + dtype.
// Returns 0 on success, < 0 if the tensor is missing.
static int get_tensor(slate_llama_t* m, const char* name,
                       const void** out_data, int* out_dtype) {
    slate_tensor_t* t = slate_gguf_get_tensor(m->meta, m->gguf, name);
    if (!t) return -1;
    *out_data = t->data;
    if (out_dtype) *out_dtype = (int)t->dtype;
    return 0;
}

slate_llama_t* slate_llama_open(slate_gguf_t* gguf) {
    if (!gguf) return NULL;

    slate_llama_t* m = (slate_llama_t*)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->gguf = gguf;
    m->meta = slate_arena_create(1 << 20);   // tensor views

    // Hyperparameters from the standard llama.* namespace.
    uint32_t n_layers = 0, d_model = 0, ffn = 0, n_heads = 0;
    uint32_t n_kv_heads = 0, max_seq = 0, vocab = 0;
    float theta_base = 10000.0f;
    float rms_eps = 1e-5f;

    if (slate_gguf_get_u32(gguf, "llama.block_count",            &n_layers) != 0 ||
        slate_gguf_get_u32(gguf, "llama.embedding_length",       &d_model)  != 0 ||
        slate_gguf_get_u32(gguf, "llama.feed_forward_length",    &ffn)      != 0 ||
        slate_gguf_get_u32(gguf, "llama.attention.head_count",   &n_heads)  != 0) {
        slate_llama_free(m);
        return NULL;
    }
    if (slate_gguf_get_u32(gguf, "llama.attention.head_count_kv", &n_kv_heads) != 0) {
        n_kv_heads = n_heads;   // MHA fallback
    }
    if (slate_gguf_get_u32(gguf, "llama.context_length", &max_seq) != 0) {
        max_seq = 2048;
    }
    if (slate_gguf_get_u32(gguf, "llama.vocab_size", &vocab) != 0) {
        // OK if absent — we'll derive it from token_embd shape below.
        vocab = 0;
    }
    (void)slate_gguf_get_f32(gguf, "llama.rope.freq_base", &theta_base);
    (void)slate_gguf_get_f32(gguf, "llama.attention.layer_norm_rms_epsilon", &rms_eps);

    m->cfg.n_layers   = (int)n_layers;
    m->cfg.d_model    = (int)d_model;
    m->cfg.ffn_hidden = (int)ffn;
    m->cfg.n_heads    = (int)n_heads;
    m->cfg.n_kv_heads = (int)n_kv_heads;
    m->cfg.head_dim   = (int)(d_model / n_heads);
    m->cfg.max_seq    = (int)max_seq;
    m->cfg.theta_base = theta_base;
    m->cfg.rms_eps    = rms_eps;
    m->cfg.vocab      = (int)vocab;

    // Global tensors.
    if (get_tensor(m, "token_embd.weight", &m->token_embd, &m->token_embd_dtype) != 0) {
        slate_llama_free(m); return NULL;
    }
    if (get_tensor(m, "output_norm.weight", &m->output_norm, NULL) != 0) {
        slate_llama_free(m); return NULL;
    }
    // output.weight may be missing if tied to token_embd.
    if (get_tensor(m, "output.weight", &m->output, &m->output_dtype) != 0) {
        m->output       = m->token_embd;
        m->output_dtype = m->token_embd_dtype;
        m->cfg.tied_output = 1;
    } else {
        m->cfg.tied_output = 0;
    }
    // Derive vocab from token_embd shape if metadata didn't carry it.
    if (m->cfg.vocab == 0) {
        slate_tensor_t* te = slate_gguf_get_tensor(m->meta, gguf, "token_embd.weight");
        if (te && te->n_dims >= 1) m->cfg.vocab = (int)te->shape[0];
    }

    // Per-block tensors.
    m->layers = (slate_llama_layer_t*)calloc((size_t)n_layers, sizeof(slate_llama_layer_t));
    if (!m->layers) { slate_llama_free(m); return NULL; }
    for (uint32_t i = 0; i < n_layers; ++i) {
        char buf[64];
        slate_llama_layer_t* lw = &m->layers[i];
        #define LOAD_REQ(field, name_fmt) do { \
            snprintf(buf, sizeof(buf), name_fmt, i); \
            if (get_tensor(m, buf, &lw->field, &lw->field##_dtype) != 0) { \
                slate_llama_free(m); return NULL; \
            } \
        } while (0)
        #define LOAD_F32(field, name_fmt) do { \
            snprintf(buf, sizeof(buf), name_fmt, i); \
            if (get_tensor(m, buf, &lw->field, NULL) != 0) { \
                slate_llama_free(m); return NULL; \
            } \
        } while (0)
        LOAD_F32(attn_norm,   "blk.%u.attn_norm.weight");
        LOAD_REQ(attn_q,      "blk.%u.attn_q.weight");
        LOAD_REQ(attn_k,      "blk.%u.attn_k.weight");
        LOAD_REQ(attn_v,      "blk.%u.attn_v.weight");
        LOAD_REQ(attn_output, "blk.%u.attn_output.weight");
        LOAD_F32(ffn_norm,    "blk.%u.ffn_norm.weight");
        LOAD_REQ(ffn_gate,    "blk.%u.ffn_gate.weight");
        LOAD_REQ(ffn_up,      "blk.%u.ffn_up.weight");
        LOAD_REQ(ffn_down,    "blk.%u.ffn_down.weight");
        #undef LOAD_REQ
        #undef LOAD_F32
    }

    return m;
}

void slate_llama_free(slate_llama_t* m) {
    if (!m) return;
    free(m->layers);
    if (m->meta) slate_arena_destroy(m->meta);
    free(m);
}

const slate_llama_config_t* slate_llama_config(const slate_llama_t* m) {
    return m ? &m->cfg : NULL;
}
const void* slate_llama_token_embd(const slate_llama_t* m) {
    return m ? m->token_embd : NULL;
}
int slate_llama_token_embd_dtype(const slate_llama_t* m) {
    return m ? m->token_embd_dtype : 0;
}
const void* slate_llama_output_norm(const slate_llama_t* m) {
    return m ? m->output_norm : NULL;
}
const void* slate_llama_output(const slate_llama_t* m) {
    return m ? m->output : NULL;
}
int slate_llama_output_dtype(const slate_llama_t* m) {
    return m ? m->output_dtype : 0;
}
const slate_llama_layer_t* slate_llama_layer(const slate_llama_t* m, int layer) {
    if (!m || layer < 0 || layer >= m->cfg.n_layers) return NULL;
    return &m->layers[layer];
}

void slate_llama_dump(const slate_llama_t* m) {
    if (!m) return;
    const slate_llama_config_t* c = &m->cfg;
    printf("=== LLaMA model ===\n");
    printf("  n_layers     = %d\n", c->n_layers);
    printf("  d_model      = %d\n", c->d_model);
    printf("  n_heads      = %d (kv_heads = %d, head_dim = %d)\n",
            c->n_heads, c->n_kv_heads, c->head_dim);
    printf("  ffn_hidden   = %d\n", c->ffn_hidden);
    printf("  max_seq      = %d\n", c->max_seq);
    printf("  vocab        = %d\n", c->vocab);
    printf("  theta_base   = %.2f   rms_eps = %.2e\n", c->theta_base, c->rms_eps);
    printf("  tied_output  = %s\n", c->tied_output ? "yes" : "no");
    printf("  token_embd_dt= %d   output_dt = %d\n",
            m->token_embd_dtype, m->output_dtype);
    for (int i = 0; i < c->n_layers && i < 4; ++i) {
        printf("  blk.%d         attn_q_dt=%d ffn_gate_dt=%d\n",
                i, m->layers[i].attn_q_dtype, m->layers[i].ffn_gate_dtype);
    }
    if (c->n_layers > 4) printf("  ... (+%d more)\n", c->n_layers - 4);
}
