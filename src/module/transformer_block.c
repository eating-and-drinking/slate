// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// transformer_block.c — pre-norm: x = x + attn(rms_norm(x));
//                                   x = x + ffn(rms_norm(x))

#include "slate/transformer.h"
#include "slate/tensor.h"
#include "slate/ops.h"
#include <stdlib.h>

typedef struct tblock {
    slate_module_t base;
    slate_module_t *norm1, *attn, *norm2, *ffn;
} tblock_t;

static slate_tensor_t* tblock_fwd(slate_module_t* self, slate_graph_ctx_t* ctx, slate_tensor_t* x) {
    tblock_t* m = (tblock_t*)self;
    slate_tensor_t* n1 = slate_module_forward(m->norm1, ctx, x);
    slate_tensor_t* a  = slate_module_forward(m->attn,  ctx, n1);
    slate_tensor_t* r1 = slate_op_add(ctx, x, a);
    slate_tensor_t* n2 = slate_module_forward(m->norm2, ctx, r1);
    slate_tensor_t* f  = slate_module_forward(m->ffn,   ctx, n2);
    slate_tensor_t* r2 = slate_op_add(ctx, r1, f);
    return r2;
}
static void tblock_reg(slate_module_t* self, slate_param_set_t* ps) {
    tblock_t* m = (tblock_t*)self;
    slate_module_register_params(m->norm1, ps);
    slate_module_register_params(m->attn,  ps);
    slate_module_register_params(m->norm2, ps);
    slate_module_register_params(m->ffn,   ps);
}
static void tblock_destroy(slate_module_t* self) {
    tblock_t* m = (tblock_t*)self;
    slate_module_destroy(m->norm1);
    slate_module_destroy(m->attn);
    slate_module_destroy(m->norm2);
    slate_module_destroy(m->ffn);
    free(m);
}

slate_module_t* slate_module_transformer_block_new(slate_arena_t* params,
                                                    int d, int h_ffn, float eps,
                                                    uint64_t seed) {
    tblock_t* m = (tblock_t*)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->base.name = "TransformerBlock";
    m->base.forward = tblock_fwd;
    m->base.register_params = tblock_reg;
    m->base.destroy = tblock_destroy;
    m->norm1 = slate_module_rmsnorm_new(params, d, eps);
    m->attn  = slate_module_attention_new(params, d, seed + 0x11);
    m->norm2 = slate_module_rmsnorm_new(params, d, eps);
    m->ffn   = slate_module_ffn_new(params, d, h_ffn, seed + 0x22);
    return &m->base;
}
