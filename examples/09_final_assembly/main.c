// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// 09_final_assembly — capstone of capstones.
//
// Demonstrates the entire Slate stack wired together end-to-end:
//   1. Open a multi-layer "mini-LLaMA" GGUF (4 layers, [64x64] Q8_0 weights)
//   2. Wrap each layer in a streamed + dequantized QuantizedLoRA module
//   3. Use AdapterManager for adapter lifecycle
//   4. Train LoRA params with gradient checkpointing
//   5. Verify the GGUF file is bit-unchanged; LoRA adapter is saved to disk
//   6. Re-open the adapter from disk and confirm it matches in-memory
//
// This is the architectural equivalent of "16 GB laptop LoRA-fine-tunes
// LLaMA-7B" in miniature form — every mechanism is the real one.

#include "slate/slate.h"
#include "slate/transformer.h"
#include "slate/gguf.h"
#include "slate/adapter_mgr.h"
#include "slate/checkpoint.h"
#include "slate/ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define N_LAYERS 4
#define DIM 64
#define RANK 4
#define BS 8
#define STEPS 100

static void fingerprint(const char* path, unsigned char out[16]) {
    FILE* fp = fopen(path, "rb"); if (!fp) { memset(out, 0, 16); return; }
    unsigned char buf[4096]; size_t n;
    uint64_t h1 = 14695981039346656037ULL, h2 = 0xCBF29CE484222325ULL;
    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0)
        for (size_t i = 0; i < n; ++i) {
            h1 = (h1 ^ buf[i]) * 1099511628211ULL;
            h2 = (h2 + buf[i]) * 16777619ULL;
        }
    fclose(fp);
    memcpy(out, &h1, 8); memcpy(out + 8, &h2, 8);
}

// Module-wrapped block: chain of N quantized-lora layers, all with shared input shape.
typedef struct chain_state {
    slate_module_t** layers;
    int n;
} chain_state_t;

static slate_tensor_t* chain_fn(slate_graph_ctx_t* ctx, slate_tensor_t* x, void* ud) {
    chain_state_t* cs = (chain_state_t*)ud;
    for (int i = 0; i < cs->n; ++i) x = slate_module_forward(cs->layers[i], ctx, x);
    return x;
}

int main(void) {
    printf("slate %s — FINAL ASSEMBLY capstone\n", SLATE_VERSION_STRING);

    unsigned char fp_before[16], fp_after[16];
    fingerprint("/tmp/mini_llama.gguf", fp_before);

    slate_gguf_t* g = slate_gguf_open("/tmp/mini_llama.gguf");
    if (!g) { puts("open mini_llama.gguf FAIL — run tools/make_mini_llama_gguf.py first"); return 1; }
    printf("[fa] GGUF: %d tensors\n", slate_gguf_n_tensors(g));

    slate_arena_t* P = slate_arena_create(16*1024*1024);
    slate_arena_t* O = slate_arena_create(16*1024*1024);
    slate_arena_t* N = slate_arena_create(16*1024*1024);
    slate_arena_t* S = slate_arena_create(64*1024*1024);

    // Wrap each GGUF tensor in a QuantizedLoRA.
    slate_module_t* layers[N_LAYERS];
    for (int i = 0; i < N_LAYERS; ++i) {
        char name[64]; snprintf(name, sizeof(name), "blk.%d.attn_q.weight", i);
        layers[i] = slate_module_quantized_lora_new(P, g, name, RANK, 4.0f, 0xC0DE + i);
        if (!layers[i]) { printf("layer %d FAIL\n", i); return 1; }
    }
    chain_state_t cs = {layers, N_LAYERS};

    slate_param_set_t ps; slate_param_set_init(&ps);
    for (int i = 0; i < N_LAYERS; ++i) slate_module_register_params(layers[i], &ps);
    int64_t total = 0;
    for (int i = 0; i < ps.n_params; ++i) total += slate_tensor_numel(ps.params[i]);
    printf("[fa] LoRA trainable params: %lld (across %d tensors)\n", (long long)total, ps.n_params);
    printf("[fa] vs base if trainable: %lld (1/%.1fx compression)\n",
           (long long)((int64_t)N_LAYERS * DIM * DIM),
           ((double)N_LAYERS * DIM * DIM) / (double)total);

    // AdapterManager
    system("rm -rf /tmp/slate_final_adapters");
    slate_adapter_mgr_t* amgr = slate_adapter_mgr_open("/tmp/slate_final_adapters");

    slate_optimizer_t* opt = slate_optimizer_adamw_new(O, &ps, 0.05f, 0.9f, 0.999f, 1e-8f, 0.0f);

    // Target: learn the identity-ish shift `y = x + 0.1`. Simple but tests
    // that the full streaming+dequant+LoRA+grad-ckpt chain learns something.
    int64_t xs[2] = {BS, DIM};
    float L0 = 0, Ln = 0;
    for (int step = 0; step < STEPS; ++step) {
        slate_graph_ctx_t ctx; slate_graph_ctx_init(&ctx, N, S); ctx.training = true;
        slate_tensor_t* x = slate_tensor_new(S, SLATE_DTYPE_F32, 2, xs, true);
        slate_tensor_t* tgt = slate_tensor_new(S, SLATE_DTYPE_F32, 2, xs, false);
        uint64_t r = 0x1000 + step;
        for (int i = 0; i < BS * DIM; ++i) {
            r = r * 6364136223846793005ULL + 1442695040888963407ULL;
            float v = (float)((double)((r >> 11) & ((1ULL<<24)-1)) / (1<<24) - 0.5) * 0.2f;
            ((float*)x->data)[i] = v;
        }
        // Compute base-chain output without LoRA contribution as target reference.
        // Simpler: use a constant target (zeros) so LoRA must drive output to 0.
        for (int i = 0; i < BS * DIM; ++i) ((float*)tgt->data)[i] = 0.0f;
        // Forward through the chain, wrapped in gradient checkpointing
        // (this is the real configuration for 7B+ LoRA training).
        slate_tensor_t* y = slate_op_checkpoint(&ctx, x, chain_fn, &cs);
        slate_tensor_t* loss = slate_op_mse_loss(&ctx, y, tgt);
        float L = ((float*)loss->data)[0];
        if (step == 0) L0 = L; if (step == STEPS - 1) Ln = L;
        if (step % 20 == 0 || step == STEPS - 1)
            printf("[fa] step %3d  loss=%.5f\n", step, L);
        slate_optimizer_zero_grad(opt);
        slate_graph_backward(&ctx, loss);
        slate_clip_grad_norm(&ps, 1.0f);
        slate_optimizer_step(opt);
        slate_graph_ctx_reset(&ctx);
    }

    // Snapshot LoRA params to a single buffer, save to adapter manager.
    size_t adapter_bytes = 0;
    for (int i = 0; i < ps.n_params; ++i)
        adapter_bytes += (size_t)slate_tensor_numel(ps.params[i]) * sizeof(float);
    void* adapter_buf = malloc(adapter_bytes);
    size_t off = 0;
    for (int i = 0; i < ps.n_params; ++i) {
        size_t n = (size_t)slate_tensor_numel(ps.params[i]) * sizeof(float);
        memcpy((char*)adapter_buf + off, ps.params[i]->data, n);
        off += n;
    }
    slate_adapter_mgr_write_candidate(amgr, adapter_buf, adapter_bytes);
    slate_adapter_mgr_promote(amgr);
    printf("[fa] adapter saved: %zu bytes\n", adapter_bytes);

    // Re-read it from disk and confirm equal.
    void* readback; size_t rb_sz;
    slate_adapter_mgr_read_current(amgr, &readback, &rb_sz);
    int adapter_match = (rb_sz == adapter_bytes && memcmp(readback, adapter_buf, adapter_bytes) == 0);
    printf("[fa] adapter byte-identical after disk round-trip: %s\n",
           adapter_match ? "yes" : "NO");

    // GGUF byte-unchanged?
    fingerprint("/tmp/mini_llama.gguf", fp_after);
    int gguf_unchanged = memcmp(fp_before, fp_after, 16) == 0;
    printf("[fa] GGUF base file unchanged (byte-level): %s\n", gguf_unchanged ? "yes" : "NO");

    printf("[fa] loss %.5f -> %.5f\n", L0, Ln);
    int loss_dropped = Ln < L0 * 0.8f;
    int ok = adapter_match && gguf_unchanged && loss_dropped;

    free(adapter_buf); free(readback);
    slate_optimizer_destroy(opt); slate_param_set_destroy(&ps);
    for (int i = 0; i < N_LAYERS; ++i) slate_module_destroy(layers[i]);
    slate_adapter_mgr_close(amgr); slate_gguf_close(g);
    slate_arena_destroy(P); slate_arena_destroy(O); slate_arena_destroy(N); slate_arena_destroy(S);
    printf("final_assembly: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
