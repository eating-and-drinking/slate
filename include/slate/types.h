// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// types.h — fundamental enums, constants, and forward declarations shared
// across every layer of the framework.

#ifndef SLATE_TYPES_H
#define SLATE_TYPES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum number of dimensions a tensor can have. Increasing this requires
// reviewing every op kernel: many of them switch on n_dims.
#define SLATE_MAX_DIMS 4

// Maximum number of inputs to a single operator graph node. The largest
// operator currently is attention (Q, K, V, mask) = 4; we allow some headroom.
#define SLATE_MAX_OP_INPUTS 6

// Element data types. The list is small on purpose; expand it only when a
// kernel actually needs the new dtype.
typedef enum slate_dtype {
    SLATE_DTYPE_F32 = 0,
    SLATE_DTYPE_F16 = 1,
    SLATE_DTYPE_BF16 = 2,
    SLATE_DTYPE_I32 = 3,
    SLATE_DTYPE_I8  = 4,
    // Block-quantized dtypes appear in M5.
    SLATE_DTYPE_Q8_0 = 16,
    SLATE_DTYPE_Q4_0 = 17,
    SLATE_DTYPE_Q4_K = 18,        // GGML "k-quant" 4.5-bit (super-block of 256)

    SLATE_DTYPE_COUNT  // sentinel
} slate_dtype_t;

// Returns the size in bytes of a single element of dtype `dt`. For block
// dtypes (Q8_0, Q4_0) this is the size of a block, not a single element.
size_t slate_dtype_size(slate_dtype_t dt);

// Returns a human-readable name like "f32" or "q4_0".
const char* slate_dtype_name(slate_dtype_t dt);

// Returns true if the dtype is a float-like dtype (F32, F16, BF16).
bool slate_dtype_is_float(slate_dtype_t dt);

// Returns true if the dtype is a block-quantized dtype.
bool slate_dtype_is_quantized(slate_dtype_t dt);

// Status / error codes. Operations that can fail return slate_status_t.
typedef enum slate_status {
    SLATE_OK = 0,
    SLATE_ERR_INVALID_ARGUMENT = -1,
    SLATE_ERR_OUT_OF_MEMORY    = -2,
    SLATE_ERR_SHAPE_MISMATCH   = -3,
    SLATE_ERR_DTYPE_MISMATCH   = -4,
    SLATE_ERR_NOT_IMPLEMENTED  = -5,
    SLATE_ERR_IO               = -6,
    SLATE_ERR_INTERNAL         = -7,
    SLATE_ERR_RUNTIME_MODE     = -8,  // wrong runtime mode for the operation
    SLATE_ERR_NO_GRAD          = -9,  // backward called without requires_grad
} slate_status_t;

// Returns a human-readable string for a status code.
const char* slate_status_string(slate_status_t s);

// Forward declarations of opaque public types. Definitions live in their own
// headers; placing the typedefs here lets unrelated headers refer to them
// without circular include problems.
typedef struct slate_arena       slate_arena_t;
typedef struct slate_threadpool  slate_threadpool_t;
typedef struct slate_tensor      slate_tensor_t;
typedef struct slate_graph       slate_graph_t;
typedef struct slate_graph_node  slate_graph_node_t;
typedef struct slate_graph_ctx   slate_graph_ctx_t;
typedef struct slate_module      slate_module_t;
typedef struct slate_param_set   slate_param_set_t;
typedef struct slate_optimizer   slate_optimizer_t;

#ifdef __cplusplus
}
#endif

#endif // SLATE_TYPES_H
