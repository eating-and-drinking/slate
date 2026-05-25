// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// causal_lm.c — full decoder-only LM:
//   token_emb + pos_emb -> N transformer blocks -> rms_norm -> lm_head
//
// Forward takes token_ids [B, T] (I32) and returns logits [B, T, V].

#include "slate/transformer.h"
#include "slate/tensor.h"
#include "slate/ops.h"
#include "slate/runtime.h"
#include <stdlib.h>
#include <string.h>

typedef struct clm {
    slate_module_t base;
    slate_module_t* tok_emb;
    slate_module_t* pos_emb;
    slate_module_t** blocks;
    int n_layers;
    slate_module_t* final_norm;
    slate_tensor_t* lm_head_W;  // [d_model, vocab]
    int max_seq, d_model, vocab;
} clm_t;

static slate_tensor_t* clm_fwd(slate_module_t* self, slate_graph_ctx_t* ctx, slate_tensor_t* token_ids) {
    clm_t* m = (clm_t*)self;
    int64_t B = token_ids->shape[0];
    int64_t T = token_ids->shape[1];

    // Build position indices [B, T]: row b is [0, 1, ..., T-1].
    int64_t ps[2] = {B, T};
    slate_tensor_t* positions = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_I32, 2, ps, false);
    int32_t* pp = (int32_t*)positions->data;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) pp[b * T + t] = (int32_t)t;
    }

    slate_tensor_t* te = slate_module_forward(m->tok_emb, ctx, token_ids);  // [B, T, D]
    slate_tensor_t* pe = slate_module_forward(m->pos_emb, ctx, positions);  // [B, T, D]
    slate_tensor_t* x = slate_op_add(ctx, te, pe);

    for (int i = 0; i < m->n_layers; ++i) {
        x = slate_module_forward(m->blocks[i], ctx, x);
    }
    x = slate_module_forward(m->final_norm, ctx, x);
    // Project to vocab: [B, T, D] @ [D, V] = [B, T, V]
    return slate_op_linear3d(ctx, x, m->lm_head_W);
}

static void clm_reg(slate_module_t* self, slate_param_set_t* ps) {
    clm_t* m = (clm_t*)self;
    slate_module_register_params(m->tok_emb, ps);
    slate_module_register_params(m->pos_emb, ps);
    for (int i = 0; i < m->n_layers; ++i) slate_module_register_params(m->blocks[i], ps);
    slate_module_register_params(m->final_norm, ps);
    slate_param_set_add(ps, m->lm_head_W);
}

static void clm_destroy(slate_module_t* self) {
    clm_t* m = (clm_t*)self;
    slate_module_destroy(m->tok_emb);
    slate_module_destroy(m->pos_emb);
    for (int i = 0; i < m->n_layers; ++i) slate_module_destroy(m->blocks[i]);
    free(m->blocks);
    slate_module_destroy(m->final_norm);
    free(m);
}

slate_module_t* slate_module_causal_lm_new(slate_arena_t* params,
                                            int vocab, int max_seq,
                                            int d_model, int n_layers,
                                            int ffn_hidden, float eps,
                                            uint64_t seed) {
    clm_t* m = (clm_t*)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->base.name = "CausalLM";
    m->base.forward = clm_fwd;
    m->base.register_params = clm_reg;
    m->base.destroy = clm_destroy;
    m->vocab = vocab; m->max_seq = max_seq;
    m->d_model = d_model; m->n_layers = n_layers;

    m->tok_emb = slate_module_embedding_new(params, vocab, d_model, seed + 1);
    m->pos_emb = slate_module_embedding_new(params, max_seq, d_model, seed + 2);
    m->blocks = (slate_module_t**)calloc((size_t)n_layers, sizeof(slate_module_t*));
    for (int i = 0; i < n_layers; ++i) {
        m->blocks[i] = slate_module_transformer_block_new(params, d_model, ffn_hidden,
                                                           eps, seed + 0x100 + i);
    }
    m->final_norm = slate_module_rmsnorm_new(params, d_model, eps);
    int64_t ws[2] = {d_model, vocab};
    m->lm_head_W = slate_tensor_new(params, SLATE_DTYPE_F32, 2, ws, true);
    // Small-stddev init for lm_head; tied embedding would also work but we keep separate.
    float* p = (float*)m->lm_head_W->data;
    int64_t n = (int64_t)d_model * vocab;
    // Simple deterministic init: small alternating values
    for (int64_t i = 0; i < n; ++i) p[i] = ((i * 2654435761u) % 1000) * 0.0002f - 0.1f;
    return &m->base;
}
