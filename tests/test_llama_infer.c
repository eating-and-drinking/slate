// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// test_llama_infer.c — end-to-end LLaMA inference test.
//
// (1) Open /tmp/slate_tiny_llama.gguf (built by make_tiny_llama_gguf.py)
// (2) Build a slate_llama_t + slate_llama_session_t
// (3) Run slate_llama_prefill on a fixed prompt
// (4) Compare last-position logits to the numpy reference from
//     tools/make_llama_ref.py within fp32 tolerance.
//
// This is the bit-precise correctness contract for slate's LLaMA
// inference path.  If it ever drifts we've changed model semantics
// silently and risked breaking real LLaMA outputs in production.

#include "slate/slate.h"
#include "slate/gguf.h"
#include "slate/llama.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    slate_gguf_t* g = slate_gguf_open("/tmp/slate_tiny_llama.gguf");
    if (!g) { puts("gguf open FAIL — run tools/make_tiny_llama_gguf.py"); return 1; }
    slate_llama_t* m = slate_llama_open(g);
    if (!m) { puts("llama_open FAIL"); return 1; }

    FILE* fp = fopen("/tmp/slate_llama_ref.bin", "rb");
    if (!fp) { puts("missing /tmp/slate_llama_ref.bin — run tools/make_llama_ref.py"); return 1; }
    int32_t prompt_len, vocab;
    fread(&prompt_len, 4, 1, fp);
    int32_t* prompt = (int32_t*)malloc((size_t)prompt_len * sizeof(int32_t));
    fread(prompt, sizeof(int32_t), prompt_len, fp);
    fread(&vocab, 4, 1, fp);
    float* ref_logits = (float*)malloc((size_t)vocab * sizeof(float));
    fread(ref_logits, sizeof(float), vocab, fp);
    fclose(fp);

    const slate_llama_config_t* cfg = slate_llama_config(m);
    printf("Loaded LLaMA: V=%d D=%d L=%d n_heads=%d head_dim=%d FFN=%d\n",
            cfg->vocab, cfg->d_model, cfg->n_layers,
            cfg->n_heads, cfg->head_dim, cfg->ffn_hidden);

    slate_llama_session_t* sess = slate_llama_session_new(m);
    if (!sess) { puts("session_new FAIL"); return 1; }

    float* out = (float*)malloc((size_t)vocab * sizeof(float));
    int rc = slate_llama_prefill(sess, prompt, prompt_len, out);
    if (rc != 0) { printf("prefill rc=%d FAIL\n", rc); return 1; }

    // Compare last-position logits.
    float linf = 0;
    double l2 = 0;
    for (int v = 0; v < vocab; ++v) {
        float d = fabsf(out[v] - ref_logits[v]);
        if (d > linf) linf = d;
        l2 += (double)d * d;
    }
    l2 = sqrt(l2);
    printf("[prefill] Linf vs numpy ref = %.6e   L2 = %.6e\n", linf, l2);
    printf("[prefill] slate out[:5]  = %.4f %.4f %.4f %.4f %.4f\n",
            out[0], out[1], out[2], out[3], out[4]);
    printf("[prefill] ref out[:5]    = %.4f %.4f %.4f %.4f %.4f\n",
            ref_logits[0], ref_logits[1], ref_logits[2], ref_logits[3], ref_logits[4]);

    int ok = (linf < 1e-3f);
    if (!ok) puts("FAIL: drift too large");

    // Sanity: position advanced
    if (slate_llama_session_position(sess) != prompt_len) {
        puts("FAIL: position not advanced correctly"); ok = 0;
    }

    free(out); free(prompt); free(ref_logits);
    slate_llama_session_free(sess);
    slate_llama_free(m);
    slate_gguf_close(g);

    printf("test_llama_infer: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
