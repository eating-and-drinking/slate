// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// tensor.h — L1: the Tensor type.
//
// A tensor is a strided multidimensional array allocated on an arena. It
// optionally has a gradient buffer (also on an arena) and a back-pointer to
// the graph node that produced it.

#ifndef SLATE_TENSOR_H
#define SLATE_TENSOR_H

#include "slate/types.h"
#include "slate/runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

struct slate_tensor {
    slate_dtype_t dtype;
    int           n_dims;
    int64_t       shape[SLATE_MAX_DIMS];
    int64_t       stride[SLATE_MAX_DIMS];  // bytes
    void*         data;                    // owned by arena
    void*         grad;                    // optional, owned by arena
    bool          requires_grad;
    bool          is_view;                 // true if a view of another tensor
    slate_graph_node_t* grad_fn;           // set by autograd during forward
};

// Construct a new contiguous tensor on `arena`. Allocates room for the data
// (and optionally the gradient buffer, if `requires_grad` is true). Returns
// NULL on out-of-memory.
slate_tensor_t* slate_tensor_new(slate_arena_t* arena,
                                 slate_dtype_t dtype,
                                 int n_dims,
                                 const int64_t* shape,
                                 bool requires_grad);

// Construct a tensor that views `source`'s buffer without owning it. Used by
// reshape/permute/transpose. The view's storage lives as long as `source`'s.
slate_tensor_t* slate_tensor_view(slate_arena_t* arena,
                                  slate_tensor_t* source,
                                  int n_dims,
                                  const int64_t* shape,
                                  const int64_t* stride);

// Returns the total number of elements (product of shape).
int64_t slate_tensor_numel(const slate_tensor_t* t);

// Returns the total number of bytes occupied by the data buffer. For
// quantized dtypes this rounds up to whole blocks.
size_t slate_tensor_nbytes(const slate_tensor_t* t);

// Returns true if the tensor is contiguous in row-major order. Some kernels
// require this.
bool slate_tensor_is_contiguous(const slate_tensor_t* t);

// Fill the data buffer with zero. Does not touch the gradient buffer.
void slate_tensor_zero(slate_tensor_t* t);

// Fill the gradient buffer with zero. No-op if the tensor has no gradient.
void slate_tensor_zero_grad(slate_tensor_t* t);

// Initialize from a row-major contiguous source buffer. Sizes must match.
slate_status_t slate_tensor_set_data(slate_tensor_t* t,
                                     const void* src,
                                     size_t src_bytes);

// Copy out to a row-major contiguous destination buffer.
slate_status_t slate_tensor_get_data(const slate_tensor_t* t,
                                     void* dst,
                                     size_t dst_bytes);

// Convenience accessor for F32 tensors. Returns NULL if dtype != F32 or out
// of bounds. Indices are in element units (not bytes).
float* slate_tensor_at_f32(slate_tensor_t* t,
                           int64_t i0, int64_t i1, int64_t i2, int64_t i3);

// Pretty-print a small tensor to stdout. Avoid for tensors with > 1024 elements.
void slate_tensor_print(const slate_tensor_t* t, const char* name);

#ifdef __cplusplus
}
#endif

#endif // SLATE_TENSOR_H
