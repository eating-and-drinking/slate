// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// module.h — L4: nn.Module analogue in C.
//
// A Module is a stateful piece of compute. It owns a set of parameter tensors
// and exposes:
//   - forward(ctx, x) -> y
//   - register_params(ps): collect parameters for the optimizer
//   - destroy(): release resources
//
// Modules compose via Sequential (or by direct calls in handwritten forwards).

#ifndef SLATE_MODULE_H
#define SLATE_MODULE_H

#include "slate/types.h"
#include "slate/tensor.h"
#include "slate/autograd.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parameter set: a dynamically-growing list of tensors. Owned by the caller;
// modules push their parameters into it.
struct slate_param_set {
    slate_tensor_t** params;
    int n_params;
    int cap_params;
};

slate_status_t slate_param_set_init(slate_param_set_t* ps);
void slate_param_set_destroy(slate_param_set_t* ps);
void slate_param_set_add(slate_param_set_t* ps, slate_tensor_t* p);

// Module vtable. Concrete modules expose a struct whose first field is
// `slate_module_t base;` and fill in these callbacks at construction.
struct slate_module {
    const char* name;  // for logging / param-naming
    slate_tensor_t* (*forward)(slate_module_t* self,
                               slate_graph_ctx_t* ctx,
                               slate_tensor_t* x);
    void (*register_params)(slate_module_t* self, slate_param_set_t* ps);
    void (*destroy)(slate_module_t* self);
};

// Convenience: call forward.
static inline slate_tensor_t* slate_module_forward(slate_module_t* m,
                                                   slate_graph_ctx_t* ctx,
                                                   slate_tensor_t* x) {
    return m->forward(m, ctx, x);
}

static inline void slate_module_register_params(slate_module_t* m,
                                                slate_param_set_t* ps) {
    if (m->register_params) m->register_params(m, ps);
}

static inline void slate_module_destroy(slate_module_t* m) {
    if (m && m->destroy) m->destroy(m);
}

// =============================================================================
// Linear: y = x @ W + b
// =============================================================================
//
// `param_arena` owns the weights and gradient buffers; it must outlive the
// module. Weights are initialized with Kaiming-uniform; bias with zeros.
slate_module_t* slate_module_linear_new(slate_arena_t* param_arena,
                                        int in_features,
                                        int out_features,
                                        bool with_bias,
                                        uint64_t init_seed);

// =============================================================================
// Sequential: chain of submodules called in order.
// =============================================================================
//
// `seq_arena` is used for the sequential's own bookkeeping; submodule
// parameters live on whatever arena the submodules were constructed with.
slate_module_t* slate_module_sequential_new(slate_arena_t* seq_arena,
                                            slate_module_t** modules,
                                            int n_modules);

#ifdef __cplusplus
}
#endif

#endif // SLATE_MODULE_H
