// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// test_llama_infer_q4k.c — verify slate runs LLaMA inference correctly
// when the linear-projection weights are stored in Q4_K_M format.
// This is the final correctness contract before slate can load a real
// llama-7b-Q4_K_M.gguf and stream tokens.

#include "slate/slate.h"
#include "slate/gguf.h"
#include "slate/llama.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    slate_gguf_t* g = slate_gguf_open("/tmp/slate_tiny_llama_q4k.gguf");
    if (!g) { puts("gguf open FAIL — run tools/make_tiny_llama_q4k_gguf.py"); return 1; }
    slate_llama_t* m = slate_llama_open(g);
    if (!m) { puts("llama_open FAIL"); return 1; }

    const slate_llama_config_t* cfg = slate_llama_config(m);
    printf("Q4_K LLaMA: V=%d D=%d L=%d n_heads=%d head_dim=%d FFN=%d\n",
            cfg->vocab, cfg->d_model, cfg->n_layers,
            cfg->n_heads, cfg->head_dim, cfg->ffn_hidden);

    // Confirm at least one weight is Q4_K (dtype enum = 18).
    const slate_llama_layer_t* L0 = slate_llama_layer(m, 0);
    printf("  L0 dtypes: attn_q=%d ffn_gate=%d (expect 18 for Q4_K)\n",
            L0->attn_q_dtype, L0->ffn_gate_dtype);
    if (L0->attn_q_dtype != 18 || L0->ffn_gate_dtype != 18) {
        puts("FAIL: expected Q4_K (dtype 18) on linear weights"); return 1;
    }

    FILE* fp = fopen("/tmp/slate_tiny_llama_q4k_ref.bin", "rb");
    if (!fp) { puts("missing ref file"); return 1; }
    int32_t prompt_len, vocab;
    fread(&prompt_len, 4, 1, fp);
    int32_t* prompt = (int32_t*)malloc((size_t)prompt_len * sizeof(int32_t));
    fread(prompt, sizeof(int32_t), prompt_len, fp);
    fread(&vocab, 4, 1, fp);
    float* ref = (float*)malloc((size_t)vocab * sizeof(float));
    fread(ref, sizeof(float), vocab, fp);
    fclose(fp);

    slate_llama_session_t* sess = slate_llama_session_new(m);
    if (!sess) { puts("session_new FAIL"); return 1; }
    float* out = (float*)malloc((size_t)vocab * sizeof(float));
    int rc = slate_llama_prefill(sess, prompt, prompt_len, out);
    if (rc != 0) { printf("prefill rc=%d FAIL\n", rc); return 1; }

    float linf = 0; double l2 = 0;
    for (int v = 0; v < vocab; ++v) {
        float d = fabsf(out[v] - ref[v]);
        if (d > linf) linf = d;
        l2 += (double)d * d;
    }
    l2 = sqrt(l2);
    printf("[prefill] Linf vs numpy ref = %.6e   L2 = %.6e\n", linf, l2);
    printf("[prefill] slate out[:5] = %.4f %.4f %.4f %.4f %.4f\n",
            out[0], out[1], out[2], out[3], out[4]);
    printf("[prefill] ref out[:5]   = %.4f %.4f %.4f %.4f %.4f\n",
            ref[0], ref[1], ref[2], ref[3], ref[4]);

    int ok = (linf < 1e-3f);
    if (!ok) puts("FAIL: drift too large");

    free(out); free(prompt); free(ref);
    slate_llama_session_free(sess);
    slate_llama_free(m);
    slate_gguf_close(g);

    printf("test_llama_infer_q4k: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
