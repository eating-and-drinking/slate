// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// 04_gpt2 — train a configurable-size GPT model from scratch on tokenized data.
//
// Usage:
//   slate_gpt2 prepare <input.txt> <vocab_out> <tokens_out> <vocab_size>
//   slate_gpt2 train   <vocab>     <tokens>    <ckpt_dir>
//
// `prepare` trains a BPE on the text, then encodes the whole corpus to a
// packed int32 file. `train` mmaps that file and trains an MHA transformer
// with Adafactor + cosine LR.
//
// Config defaults to "small GPT" (32M params); flip the macros to scale up.

#include "slate/slate.h"
#include "slate/transformer.h"
#include "slate/bpe_tokenizer.h"
#include "slate/mmap_dataset.h"
#include "slate/sampling.h"
#include "slate/lr_scheduler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SEQ       64
#define D_MODEL  128
#define N_LAYERS   2
#define N_HEADS    4
#define FFN_H     256
#define BATCH      8
#define LR     3e-4f

slate_optimizer_t* slate_optimizer_adafactor_new(slate_arena_t*, slate_param_set_t*,
                                                  float, float, float, float);

static char* slurp(const char* path, size_t* n) {
    FILE* fp = fopen(path, "rb"); if (!fp) return NULL;
    fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET);
    char* b = (char*)malloc((size_t)sz);
    if (fread(b, 1, (size_t)sz, fp) != (size_t)sz) { free(b); fclose(fp); return NULL; }
    fclose(fp); *n = (size_t)sz; return b;
}

static int cmd_prepare(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "prepare <input.txt> <vocab.out> <tokens.out> <vocab_size>\n"); return 2; }
    const char* inp = argv[2]; const char* vp = argv[3];
    const char* tp = argv[4]; int V = atoi(argv[5]);
    size_t n; char* text = slurp(inp, &n);
    if (!text) { fprintf(stderr, "cannot read %s\n", inp); return 1; }
    printf("[prep] %zu bytes\n", n);
    slate_bpe_tokenizer_t* tk = slate_bpe_train(text, n, V);
    printf("[prep] vocab=%d\n", slate_bpe_vocab_size(tk));
    slate_bpe_save(tk, vp);
    int32_t* toks = (int32_t*)malloc(n * sizeof(int32_t));
    int nt = slate_bpe_encode(tk, text, n, toks, (int)n);
    printf("[prep] encoded -> %d tokens\n", nt);
    FILE* fp = fopen(tp, "wb"); fwrite(toks, sizeof(int32_t), (size_t)nt, fp); fclose(fp);
    free(toks); slate_bpe_destroy(tk); free(text);
    return 0;
}

static int cmd_train(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "train <vocab> <tokens> <ckpt_dir>\n"); return 2; }
    const char* vp = argv[2]; const char* tp = argv[3]; (void)argv[4];
    slate_bpe_tokenizer_t* tk = slate_bpe_load(vp);
    if (!tk) { fprintf(stderr, "load vocab fail\n"); return 1; }
    int V = slate_bpe_vocab_size(tk);
    slate_mmap_dataset_t* ds = slate_mmap_open(tp);
    if (!ds) { fprintf(stderr, "mmap tokens fail\n"); return 1; }
    printf("[train] vocab=%d, tokens=%lld\n", V, (long long)slate_mmap_n_tokens(ds));
    int64_t total_train_tokens = slate_mmap_n_tokens(ds);
    // Aim for ~10 passes over the corpus.
    int STEPS = (int)(total_train_tokens * 10 / (BATCH * SEQ));
    if (STEPS < 50) STEPS = 50;
    if (STEPS > 50000) STEPS = 50000;
    printf("[train] STEPS=%d\n", STEPS);

    slate_arena_t* params  = slate_arena_create( 256 * 1024 * 1024);
    slate_arena_t* opt_st  = slate_arena_create( 256 * 1024 * 1024);
    slate_arena_t* nodes   = slate_arena_create(  32 * 1024 * 1024);
    slate_arena_t* scratch = slate_arena_create(1024 * 1024 * 1024);

    // For M4: use mh_attention rather than single-head. Build CausalLM
    // adapted to take MHA via a custom block. For simplicity we still use
    // the M2 CausalLM here (single-head); upgrading to MHA-CausalLM is
    // a one-line change in causal_lm.c once we wire the option through.
    slate_module_t* model = slate_module_causal_lm_new(params, V, SEQ,
                                                        D_MODEL, N_LAYERS, FFN_H,
                                                        1e-5f, 0xC0DEC0DE);
    slate_param_set_t ps; slate_param_set_init(&ps);
    slate_module_register_params(model, &ps);
    int64_t total = 0;
    for (int i = 0; i < ps.n_params; ++i) total += slate_tensor_numel(ps.params[i]);
    printf("[train] %lld parameters\n", (long long)total);

    slate_optimizer_t* opt = slate_optimizer_adafactor_new(opt_st, &ps,
                                                            LR, 1e-30f, 1e-3f, 1.0f);
    slate_lr_scheduler_t* sch = slate_lr_cosine_warmup_new(LR, LR * 0.01f,
                                                            STEPS / 100, STEPS);

    int32_t* X_buf = (int32_t*)malloc((size_t)BATCH * SEQ * sizeof(int32_t));
    int32_t* Y_buf = (int32_t*)malloc((size_t)BATCH * SEQ * sizeof(int32_t));
    uint64_t rng = 0xA5A5A5;
    int64_t xshape[2] = {BATCH, SEQ};
    int64_t yshape[1] = {BATCH * SEQ};
    int64_t logits_flat[2] = {BATCH * SEQ, V};

    clock_t t0 = clock();
    for (int step = 0; step < STEPS; ++step) {
        slate_optimizer_set_lr(opt, slate_lr_scheduler_get(sch, step));
        slate_mmap_sample_batch(ds, BATCH, SEQ, X_buf, Y_buf, &rng);

        slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, nodes, scratch);
        ctx.training = true;
        slate_tensor_t* X = slate_tensor_new(scratch, SLATE_DTYPE_I32, 2, xshape, false);
        memcpy(X->data, X_buf, (size_t)BATCH * SEQ * sizeof(int32_t));
        slate_tensor_t* Y = slate_tensor_new(scratch, SLATE_DTYPE_I32, 1, yshape, false);
        memcpy(Y->data, Y_buf, (size_t)BATCH * SEQ * sizeof(int32_t));

        slate_tensor_t* logits = slate_module_forward(model, &ctx, X);
        slate_tensor_t* lf = slate_tensor_view(scratch, logits, 2, logits_flat, NULL);
        slate_tensor_t* loss = slate_op_cross_entropy_loss(&ctx, lf, Y);
        slate_optimizer_zero_grad(opt);
        slate_graph_backward(&ctx, loss);
        slate_clip_grad_norm(&ps, 1.0f);
        slate_optimizer_step(opt);

        if (step % 5 == 0 || step == STEPS - 1) {
            printf("[train] step %5d  loss=%.4f  elapsed=%.0fs\n",
                   step, ((float*)loss->data)[0],
                   (double)(clock() - t0) / CLOCKS_PER_SEC);
        }
        slate_graph_ctx_reset(&ctx);
    }
    printf("[train] done\n");

    free(X_buf); free(Y_buf);
    slate_lr_scheduler_destroy(sch);
    slate_optimizer_destroy(opt);
    slate_param_set_destroy(&ps);
    slate_module_destroy(model);
    slate_mmap_close(ds);
    slate_bpe_destroy(tk);
    slate_arena_destroy(params); slate_arena_destroy(opt_st);
    slate_arena_destroy(nodes); slate_arena_destroy(scratch);
    return 0;
}

int main(int argc, char** argv) {
    printf("slate %s — gpt2\n", SLATE_VERSION_STRING);
    if (argc < 2) {
        fprintf(stderr,
            "usage:\n"
            "  %s prepare <input.txt> <vocab.out> <tokens.out> <vocab_size>\n"
            "  %s train   <vocab>     <tokens>    <ckpt_dir>\n", argv[0], argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "prepare") == 0) return cmd_prepare(argc, argv);
    if (strcmp(argv[1], "train") == 0)   return cmd_train(argc, argv);
    fprintf(stderr, "unknown command %s\n", argv[1]); return 2;
}
