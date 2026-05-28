// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// test_llama_load.c — verify slate_llama_open parses the standard
// llama.cpp GGUF schema correctly: hyperparameters from kv metadata,
// per-layer + global weight tensors by name, tied-output detection.

#include "slate/slate.h"
#include "slate/gguf.h"
#include "slate/llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    slate_gguf_t* g = slate_gguf_open("/tmp/slate_tiny_llama.gguf");
    if (!g) { puts("gguf open FAIL — run tools/make_tiny_llama_gguf.py first"); return 1; }
    printf("gguf opened: %d tensors\n", slate_gguf_n_tensors(g));

    slate_llama_t* m = slate_llama_open(g);
    if (!m) { puts("slate_llama_open FAIL"); return 1; }

    slate_llama_dump(m);
    const slate_llama_config_t* c = slate_llama_config(m);
    int ok = 1;

    // Verify config matches what tools/make_tiny_llama_gguf.py emitted.
    if (c->n_layers   != 2)     { puts("FAIL: n_layers");   ok = 0; }
    if (c->d_model    != 32)    { puts("FAIL: d_model");    ok = 0; }
    if (c->n_heads    != 2)     { puts("FAIL: n_heads");    ok = 0; }
    if (c->n_kv_heads != 2)     { puts("FAIL: n_kv_heads"); ok = 0; }
    if (c->head_dim   != 16)    { puts("FAIL: head_dim");   ok = 0; }
    if (c->ffn_hidden != 64)    { puts("FAIL: ffn_hidden"); ok = 0; }
    if (c->vocab      != 64)    { puts("FAIL: vocab");      ok = 0; }
    if (c->max_seq    != 128)   { puts("FAIL: max_seq");    ok = 0; }
    if (c->tied_output != 0)    { puts("FAIL: tied (this fixture is untied)"); ok = 0; }
    if (c->theta_base < 9999.0f || c->theta_base > 10001.0f) {
        puts("FAIL: theta_base"); ok = 0;
    }

    // Verify all required global tensors resolved.
    if (!slate_llama_token_embd(m))  { puts("FAIL: token_embd");  ok = 0; }
    if (!slate_llama_output_norm(m)) { puts("FAIL: output_norm"); ok = 0; }
    if (!slate_llama_output(m))      { puts("FAIL: output");      ok = 0; }

    // Verify per-block tensors resolved.
    for (int i = 0; i < c->n_layers; ++i) {
        const slate_llama_layer_t* L = slate_llama_layer(m, i);
        if (!L)                  { puts("FAIL: layer slot");  ok = 0; continue; }
        if (!L->attn_norm)       { puts("FAIL: attn_norm");   ok = 0; }
        if (!L->attn_q)          { puts("FAIL: attn_q");      ok = 0; }
        if (!L->attn_k)          { puts("FAIL: attn_k");      ok = 0; }
        if (!L->attn_v)          { puts("FAIL: attn_v");      ok = 0; }
        if (!L->attn_output)     { puts("FAIL: attn_output"); ok = 0; }
        if (!L->ffn_norm)        { puts("FAIL: ffn_norm");    ok = 0; }
        if (!L->ffn_gate)        { puts("FAIL: ffn_gate");    ok = 0; }
        if (!L->ffn_up)          { puts("FAIL: ffn_up");      ok = 0; }
        if (!L->ffn_down)        { puts("FAIL: ffn_down");    ok = 0; }
        // Sanity check: f32 dtype enum is 0
        if (L->attn_q_dtype != 0) { puts("FAIL: attn_q expected f32"); ok = 0; }
    }

    // The first token_embd row should be the first 32 floats of the data
    // block.  Sample one value through the pointer.
    const float* te = (const float*)slate_llama_token_embd(m);
    if (te) {
        printf("token_embd[0,0..3] = %.4f %.4f %.4f %.4f\n",
                te[0], te[1], te[2], te[3]);
    }

    slate_llama_free(m);
    slate_gguf_close(g);
    printf("test_llama_load: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
