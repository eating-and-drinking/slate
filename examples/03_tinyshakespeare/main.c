// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// 03_tinyshakespeare — train a char-level mini-GPT on the tinyshakespeare corpus.
//
// Usage:
//   slate_tinyshakespeare <path/to/input.txt>
//
// Get the corpus from:
//   https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
//
// Architecture (~1M params): single-head, learned positional embedding,
// pre-norm RMSNorm, SwiGLU FFN.

#include "slate/slate.h"
#include "slate/transformer.h"
#include "slate/char_tokenizer.h"
#include "slate/lr_scheduler.h"
#include "slate/data_simple.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SEQ      128
#define D_MODEL  128
#define N_LAYERS  4
#define FFN_H    256
#define BATCH     16
#define STEPS    5000
#define LR     5e-4f
#define GEN_TOKENS 200

static char* slurp(const char* path, size_t* out_n) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END); long n = ftell(fp); fseek(fp, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)n);
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) { free(buf); fclose(fp); return NULL; }
    fclose(fp);
    *out_n = (size_t)n;
    return buf;
}

static int argmax(const float* p, int n) {
    int best = 0; float bv = p[0];
    for (int i = 1; i < n; ++i) if (p[i] > bv) { bv = p[i]; best = i; }
    return best;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <input.txt>\n", argv[0]); return 2; }
    printf("slate %s — tinyshakespeare\n", SLATE_VERSION_STRING);
    size_t n_text = 0;
    char* text = slurp(argv[1], &n_text);
    if (!text) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    printf("[ts] %zu chars in corpus\n", n_text);

    slate_char_tokenizer_t* tk = slate_char_tokenizer_build(text, n_text);
    int V = slate_char_tokenizer_vocab_size(tk);
    printf("[ts] vocab=%d\n", V);

    int32_t* tokens = (int32_t*)malloc(n_text * sizeof(int32_t));
    int n_tok = slate_char_tokenizer_encode(tk, text, n_text, tokens, (int)n_text);
    int n_train = (int)(n_tok * 0.9);
    int n_val   = n_tok - n_train;
    printf("[ts] train=%d val=%d tokens\n", n_train, n_val);

    slate_arena_t* params  = slate_arena_create( 64 * 1024 * 1024);
    slate_arena_t* opt_st  = slate_arena_create(128 * 1024 * 1024);
    slate_arena_t* nodes   = slate_arena_create( 32 * 1024 * 1024);
    slate_arena_t* scratch = slate_arena_create(256 * 1024 * 1024);

    slate_module_t* model = slate_module_causal_lm_new(params, V, SEQ,
                                                        D_MODEL, N_LAYERS, FFN_H,
                                                        1e-5f, 0xC0DEC0DEULL);
    slate_param_set_t ps; slate_param_set_init(&ps);
    slate_module_register_params(model, &ps);
    int64_t total = 0;
    for (int i = 0; i < ps.n_params; ++i) total += slate_tensor_numel(ps.params[i]);
    printf("[ts] %lld params (%d tensors)\n", (long long)total, ps.n_params);

    slate_optimizer_t* opt = slate_optimizer_adamw_new(opt_st, &ps,
                                                       LR, 0.9f, 0.95f,
                                                       1e-8f, 0.1f);
    slate_lr_scheduler_t* sch = slate_lr_cosine_warmup_new(LR, LR * 0.01f,
                                                            STEPS / 100, STEPS);

    int64_t xshape[2] = {BATCH, SEQ};
    int64_t yshape[1] = {BATCH * SEQ};
    int64_t logits_flat[2] = {BATCH * SEQ, V};

    clock_t t0 = clock();
    uint64_t rng = 0xA5A5;
    for (int step = 0; step < STEPS; ++step) {
        slate_optimizer_set_lr(opt, slate_lr_scheduler_get(sch, step));
        slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, nodes, scratch);
        ctx.training = true;

        slate_tensor_t* X = slate_tensor_new(scratch, SLATE_DTYPE_I32, 2, xshape, false);
        slate_tensor_t* Y = slate_tensor_new(scratch, SLATE_DTYPE_I32, 1, yshape, false);
        int32_t* xp = (int32_t*)X->data;
        int32_t* yp = (int32_t*)Y->data;
        for (int b = 0; b < BATCH; ++b) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            int start = (int)((rng >> 11) % (uint64_t)(n_train - SEQ - 1));
            for (int t = 0; t < SEQ; ++t) {
                xp[b * SEQ + t] = tokens[start + t];
                yp[b * SEQ + t] = tokens[start + t + 1];
            }
        }
        slate_tensor_t* logits = slate_module_forward(model, &ctx, X);
        slate_tensor_t* lf = slate_tensor_view(scratch, logits, 2, logits_flat, NULL);
        slate_tensor_t* loss = slate_op_cross_entropy_loss(&ctx, lf, Y);
        slate_optimizer_zero_grad(opt);
        slate_graph_backward(&ctx, loss);
        slate_clip_grad_norm(&ps, 1.0f);
        slate_optimizer_step(opt);
        if (step % 100 == 0 || step == STEPS - 1) {
            printf("[ts] step %5d  loss=%.4f  elapsed=%.0fs\n",
                   step, ((float*)loss->data)[0],
                   (double)(clock() - t0) / CLOCKS_PER_SEC);
        }
        slate_graph_ctx_reset(&ctx);
    }

    // Greedy generation from a fixed prompt.
    const char* prompt = "ROMEO:";
    int p_len = (int)strlen(prompt);
    int32_t ctx_buf[SEQ]; for (int i = 0; i < SEQ; ++i) ctx_buf[i] = 0;
    int pn = slate_char_tokenizer_encode(tk, prompt, (size_t)p_len, ctx_buf, SEQ);
    char out[GEN_TOKENS + 16]; memcpy(out, prompt, (size_t)p_len);
    int n_out = p_len;
    int64_t gen_xshape[2] = {1, SEQ};

    for (int g = 0; g < GEN_TOKENS; ++g) {
        slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, nodes, scratch);
        ctx.training = false;
        slate_tensor_t* X = slate_tensor_new(scratch, SLATE_DTYPE_I32, 2, gen_xshape, false);
        int32_t* xp = (int32_t*)X->data;
        // Right-align prompt in a SEQ window.
        int start = (pn > SEQ) ? pn - SEQ : 0;
        int eff = pn - start;
        for (int i = 0; i < SEQ - eff; ++i) xp[i] = 0;
        for (int i = 0; i < eff; ++i) xp[SEQ - eff + i] = ctx_buf[start + i];
        slate_tensor_t* logits = slate_module_forward(model, &ctx, X);
        const float* p_last = (const float*)logits->data + (int64_t)0 * SEQ * V + (int64_t)(SEQ - 1) * V;
        int next = argmax(p_last, V);
        if (pn < SEQ) ctx_buf[pn] = (int32_t)next;
        else { memmove(ctx_buf, ctx_buf + 1, (SEQ - 1) * sizeof(int32_t)); ctx_buf[SEQ - 1] = (int32_t)next; }
        pn = (pn < SEQ) ? pn + 1 : SEQ;
        char c; slate_char_tokenizer_decode(tk, &(int32_t){next}, 1, &c, 1);
        if (n_out < (int)sizeof(out) - 1) out[n_out++] = c;
        slate_graph_ctx_reset(&ctx);
    }
    out[n_out] = 0;
    printf("[ts] sample:\n%s\n", out);

    free(tokens); free(text);
    slate_char_tokenizer_destroy(tk);
    slate_lr_scheduler_destroy(sch);
    slate_optimizer_destroy(opt);
    slate_param_set_destroy(&ps);
    slate_module_destroy(model);
    slate_arena_destroy(params); slate_arena_destroy(opt_st);
    slate_arena_destroy(nodes); slate_arena_destroy(scratch);
    return 0;
}
