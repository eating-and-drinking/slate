#ifndef SLATE_OPS_H
#define SLATE_OPS_H
#include "slate/types.h"
#include "slate/autograd.h"
#ifdef __cplusplus
extern "C" {
#endif
slate_tensor_t* slate_op_matmul(slate_graph_ctx_t*, slate_tensor_t*, slate_tensor_t*);
slate_tensor_t* slate_op_add(slate_graph_ctx_t*, slate_tensor_t*, slate_tensor_t*);
slate_tensor_t* slate_op_mul(slate_graph_ctx_t*, slate_tensor_t*, slate_tensor_t*);
slate_tensor_t* slate_op_relu(slate_graph_ctx_t*, slate_tensor_t*);
slate_tensor_t* slate_op_sigmoid(slate_graph_ctx_t*, slate_tensor_t*);
slate_tensor_t* slate_op_silu(slate_graph_ctx_t*, slate_tensor_t*);
slate_tensor_t* slate_op_softmax(slate_graph_ctx_t*, slate_tensor_t*);
slate_tensor_t* slate_op_scale(slate_graph_ctx_t*, slate_tensor_t*, float);
slate_tensor_t* slate_op_transpose_last2(slate_graph_ctx_t*, slate_tensor_t*);
slate_tensor_t* slate_op_rms_norm(slate_graph_ctx_t*, slate_tensor_t*, slate_tensor_t*, float);
slate_tensor_t* slate_op_embedding(slate_graph_ctx_t*, slate_tensor_t*, slate_tensor_t*);
slate_tensor_t* slate_op_bmm(slate_graph_ctx_t*, slate_tensor_t*, slate_tensor_t*);
slate_tensor_t* slate_op_causal_mask(slate_graph_ctx_t*, slate_tensor_t*, float);
slate_tensor_t* slate_op_permute_12(slate_graph_ctx_t*, slate_tensor_t*);
slate_tensor_t* slate_op_mse_loss(slate_graph_ctx_t*, slate_tensor_t*, slate_tensor_t*);
slate_tensor_t* slate_op_cross_entropy_loss(slate_graph_ctx_t*, slate_tensor_t*, slate_tensor_t*);
slate_tensor_t* slate_op_add_bias(slate_graph_ctx_t*, slate_tensor_t*, slate_tensor_t*);

// RoPE (Rotary Position Embedding) — LLaMA / Mistral / etc. style.
// Rotates pairs of dimensions of the input by per-position frequencies
// derived from `theta_base` (10000 for LLaMA).
//   x         : [B, T, n_heads, head_dim] f32, head_dim must be EVEN
//   positions : [T] i32, position index for each token
//   theta_base: base frequency (10000.0f for LLaMA)
//   start_pos : if >= 0, override positions and use [start_pos, start_pos+1, ...]
//               (convenient for inference where positions are trivially contiguous;
//               pass < 0 to use the `positions` tensor)
// Returns y with the same shape as x, with the standard non-interleaved
// rotation applied to the first/second halves of head_dim.
slate_tensor_t* slate_op_rope(slate_graph_ctx_t* ctx,
                               slate_tensor_t* x,
                               slate_tensor_t* positions,
                               float theta_base);
#ifdef __cplusplus
}
#endif
#endif
