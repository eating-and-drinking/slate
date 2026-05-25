// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#define _POSIX_C_SOURCE 200809L
#include "slate/streaming_module.h"
#include "slate/transformer.h"  // for slate_op_linear3d
#include "slate/stream_io.h"
#include "slate/tensor.h"
#include "slate/ops.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

static atomic_size_t g_currently_resident = 0;
static atomic_size_t g_peak_resident = 0;

static void track_load(size_t bytes) {
    size_t cur = atomic_fetch_add(&g_currently_resident, bytes) + bytes;
    size_t prev = atomic_load(&g_peak_resident);
    while (cur > prev && !atomic_compare_exchange_weak(&g_peak_resident, &prev, cur)) {}
}
static void track_evict(size_t bytes) { atomic_fetch_sub(&g_currently_resident, bytes); }

size_t slate_streaming_peak_bytes(void) { return atomic_load(&g_peak_resident); }
void slate_streaming_reset_peak(void) {
    atomic_store(&g_peak_resident, atomic_load(&g_currently_resident));
}

typedef struct sl {
    slate_module_t base;
    char* path;
    int in_features, out_features;
} sl_t;

static slate_tensor_t* sl_fwd(slate_module_t* self, slate_graph_ctx_t* ctx, slate_tensor_t* x) {
    sl_t* m = (sl_t*)self;
    // mmap weights now
    slate_tensor_t* W = slate_stream_mmap(ctx->scratch_arena, m->path);
    if (!W) return NULL;
    size_t resident = slate_stream_resident_bytes(W);
    track_load(resident);

    // Compute y = x @ W using existing ops. We need a real (non-mmapped)
    // copy of W for the gradient buffer not to be needed; mmaps are PROT_READ
    // so matmul will not write into them. Output is on scratch.
    slate_tensor_t* y;
    if (x->n_dims == 2) y = slate_op_matmul(ctx, x, W);
    else                y = slate_op_linear3d(ctx, x, W);

    // For M5.2 we evict immediately after the FORWARD result is computed.
    // (In a real training run we'd hold until backward; that's M5.3 with
    // RuntimeMode-aware StreamUnit lifecycle.)
    slate_stream_release(W);
    track_evict(resident);
    return y;
}

static void sl_reg(slate_module_t* self, slate_param_set_t* ps) {
    (void)self; (void)ps;
    // No trainable params (streaming base weights are frozen by definition).
}
static void sl_destroy(slate_module_t* self) {
    sl_t* m = (sl_t*)self;
    free(m->path);
    free(m);
}

slate_module_t* slate_module_streaming_linear_new(const char* path,
                                                   int in_features,
                                                   int out_features) {
    sl_t* m = (sl_t*)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->base.name = "StreamingLinear";
    m->base.forward = sl_fwd;
    m->base.register_params = sl_reg;
    m->base.destroy = sl_destroy;
    m->path = strdup(path);
    m->in_features = in_features;
    m->out_features = out_features;
    return &m->base;
}
