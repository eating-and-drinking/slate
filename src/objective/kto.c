// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#include "slate/kto.h"
#include "slate/dpo.h"
#include "slate/tensor.h"
#include "slate/autograd.h"
#include "slate/runtime.h"
#include <math.h>
#include <string.h>

// We rely on slate_op_seq_logp to get policy logp[B], then build KTO loss
// on top of those + reference logps.

typedef struct {
    float* delta;       // [B]: signed r_θ (negative for bad samples)
    int32_t* labels;
    float w_good, w_bad;
    int64_t B;
} kto_state_t;

static void kto_backward(slate_graph_node_t* node) {
    slate_tensor_t* logp_p = node->inputs[0];  // policy logp
    slate_tensor_t* out = node->output;
    kto_state_t* st = (kto_state_t*)node->user_data;
    if (!logp_p->requires_grad || !logp_p->grad) return;
    float dy = ((const float*)out->grad)[0];
    float scale = dy / (float)st->B;
    float* dp = (float*)logp_p->grad;
    // For good (label = +1):  L = σ(-r)        d L / d r = -σ(-r)(1-σ(-r))*[derivative wrt r]
    //                                          d/dr σ(-r) = -σ(-r)*(1-σ(-r))
    //   so d L / d r = -σ(-r)(1-σ(-r))
    // For bad (label = -1):   L = σ(r)         d L / d r = σ(r)(1-σ(r))
    //
    // r = β * (logp_p - logp_ref); d r / d logp_p = β  (logp_ref is a constant)
    //
    // Combined: d L / d logp_p = β * sign * σ(z)(1-σ(z))
    //   where for good: z = -r, sign = -1  → dL/d logp_p = -β σ(-r)(1-σ(-r))
    //   for bad:        z =  r, sign = +1  → dL/d logp_p =  β σ( r)(1-σ( r))
    //
    // Multiply per-sample weight (desirable_weight or undesirable_weight) and scale.
    for (int64_t b = 0; b < st->B; ++b) {
        float r = st->delta[b];
        if (st->labels[b] > 0) {
            float z = -r;
            float sig = (z >= 0) ? 1.0f/(1.0f+expf(-z)) : expf(z)/(1.0f+expf(z));
            float w = st->w_good;
            dp[b] += -1.0f * w * sig * (1.0f - sig) * (-1.0f) * scale;
            // The two negatives cancel: dp[b] += w * sig*(1-sig) * scale (×β handled below)
            // (Actually we still need β factor.)
        }
        // Wait: we factored out scale already. Re-derive cleanly:
        // For each sample, the per-sample loss is L_b and total = (1/B) sum_b w_b L_b.
        // We've absorbed w_b inline. We've factored 1/B into `scale = dy/B`.
        // The β factor from d r / d logp_p still needs to apply.
    }
    // Clearer second pass: redo cleanly without inline arithmetic confusion.
    memset(NULL, 0, 0); // (no-op placeholder; gradients already applied above)
    (void)dy; (void)scale;
}

// Re-implement KTO loss clean: build it as DPO-style δ for each sample,
// reuse seq_logp for the diff, but with per-sample direction based on label.
//
// Simpler approach: just inline the whole thing. For samples with label=+1:
//   loss_b = w_good * σ(-r_b)
// For label=-1:
//   loss_b = w_bad  * σ( r_b)
// And total = mean over batch.
//
// d/d r_b L_b = (label=+1) → -w_good * σ(-r) * (1 - σ(-r))
//              = -w_good * σ(-r) * σ(r)
// d/d r_b L_b = (label=-1) →  w_bad  * σ( r) * (1 - σ( r))
//              =  w_bad  * σ(r) * σ(-r)
// d r_b / d logp_p = β
// So total: d L / d logp_p[b] = β/B * (per-sample expression above)

static void kto_backward_v2(slate_graph_node_t* node) {
    slate_tensor_t* logp_p = node->inputs[0];
    slate_tensor_t* out = node->output;
    kto_state_t* st = (kto_state_t*)node->user_data;
    if (!logp_p->requires_grad || !logp_p->grad) return;
    float dy = ((const float*)out->grad)[0];
    float scale = dy / (float)st->B;
    float* dp = (float*)logp_p->grad;
    for (int64_t b = 0; b < st->B; ++b) {
        float r = st->delta[b];
        float sig_r  = (r >= 0) ? 1.0f/(1.0f+expf(-r)) : expf(r)/(1.0f+expf(r));
        float sig_nr = 1.0f - sig_r;
        if (st->labels[b] > 0) {
            // loss_b = w_good * σ(-r); dL/dr = -w_good * σ(-r) * σ(r)
            dp[b] += -st->w_good * sig_nr * sig_r * scale;  // (×β below)
        } else {
            // loss_b = w_bad * σ(r); dL/dr = w_bad * σ(r) * σ(-r)
            dp[b] +=  st->w_bad  * sig_r * sig_nr * scale;
        }
    }
    // Beta gets multiplied via δ relationship: r = β * (logp_p - logp_ref).
    // We need to multiply the whole gradient by β.
    // It's cleaner to pre-multiply when constructing delta, but we kept beta
    // separate. Apply it post-hoc:
    // (Actually all I need is to multiply each dp[b] by β. Let me re-do.)
    // We already stored r = β*(logp_p - logp_ref) in delta. So d r / d logp_p = β.
    // The chain rule says dL/d logp_p = (dL/d r) * β.
    // So each dp[b] should be multiplied by β. But we haven't done that yet.
    // Store β in state... actually for the toy test, the test uses β=0.1 so
    // it's just a constant scaling. Let me apply it properly:
}

// Use _v2 and add beta multiplier.
typedef struct {
    float* r_arr;
    int32_t* labels;
    float w_good, w_bad, beta;
    int64_t B;
} kto_state2_t;

static void kto_back_real(slate_graph_node_t* node) {
    slate_tensor_t* logp_p = node->inputs[0];
    slate_tensor_t* out = node->output;
    kto_state2_t* st = (kto_state2_t*)node->user_data;
    if (!logp_p->requires_grad || !logp_p->grad) return;
    float dy = ((const float*)out->grad)[0];
    float scale = (dy / (float)st->B) * st->beta;
    float* dp = (float*)logp_p->grad;
    for (int64_t b = 0; b < st->B; ++b) {
        float r = st->r_arr[b];
        float sig_r  = (r >= 0) ? 1.0f/(1.0f+expf(-r)) : expf(r)/(1.0f+expf(r));
        float sig_nr = 1.0f - sig_r;
        if (st->labels[b] > 0) dp[b] += -st->w_good * sig_nr * sig_r * scale;
        else                    dp[b] +=  st->w_bad  * sig_r  * sig_nr * scale;
    }
    (void)kto_backward; (void)kto_backward_v2;
}

slate_tensor_t* slate_op_kto_loss(slate_graph_ctx_t* ctx,
                                   slate_tensor_t* logits,
                                   slate_tensor_t* targets,
                                   slate_tensor_t* logps_ref,
                                   slate_tensor_t* labels,
                                   float beta,
                                   float wg, float wb) {
    slate_tensor_t* logp_p = slate_op_seq_logp(ctx, logits, targets);
    int64_t B = logp_p->shape[0];
    int64_t s1[1] = {1};
    slate_tensor_t* out = slate_tensor_new(ctx->scratch_arena, SLATE_DTYPE_F32, 1, s1, false);
    float* pp = (float*)logp_p->data;
    float* pr = (float*)logps_ref->data;
    int32_t* lbl = (int32_t*)labels->data;
    float* r_arr = (float*)slate_arena_alloc(ctx->scratch_arena, (size_t)B*sizeof(float), 16);
    double L = 0;
    for (int64_t b = 0; b < B; ++b) {
        float r = beta * (pp[b] - pr[b]);
        r_arr[b] = r;
        float sig;
        if (lbl[b] > 0) { float z = -r; sig = (z>=0)?1/(1+expf(-z)):expf(z)/(1+expf(z)); L += wg * sig; }
        else            { float z =  r; sig = (z>=0)?1/(1+expf(-z)):expf(z)/(1+expf(z)); L += wb * sig; }
    }
    ((float*)out->data)[0] = (float)(L / (double)B);
    slate_tensor_t* inputs[1] = {logp_p};
    slate_graph_node_t* node = slate_graph_record(ctx, "kto_loss", inputs, 1, out, kto_back_real);
    if (node) {
        kto_state2_t* st = (kto_state2_t*)slate_arena_alloc(ctx->scratch_arena, sizeof(*st), 16);
        int32_t* lcopy = (int32_t*)slate_arena_alloc(ctx->scratch_arena, (size_t)B*sizeof(int32_t), 16);
        memcpy(lcopy, lbl, (size_t)B*sizeof(int32_t));
        st->r_arr = r_arr; st->labels = lcopy;
        st->w_good = wg; st->w_bad = wb; st->beta = beta; st->B = B;
        node->user_data = st;
    }
    if (!out->grad) out->grad = slate_arena_alloc(ctx->scratch_arena, sizeof(float), 16);
    out->requires_grad = (node != NULL) || logits->requires_grad;
    return out;
}
