// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// module.c — parameter set bookkeeping.

#include "slate/module.h"

#include <stdlib.h>
#include <string.h>

slate_status_t slate_param_set_init(slate_param_set_t* ps) {
    if (!ps) return SLATE_ERR_INVALID_ARGUMENT;
    ps->params = NULL;
    ps->n_params = 0;
    ps->cap_params = 0;
    return SLATE_OK;
}

void slate_param_set_destroy(slate_param_set_t* ps) {
    if (!ps) return;
    free(ps->params);
    ps->params = NULL;
    ps->n_params = 0;
    ps->cap_params = 0;
}

void slate_param_set_add(slate_param_set_t* ps, slate_tensor_t* p) {
    if (!ps || !p) return;
    if (ps->n_params >= ps->cap_params) {
        int new_cap = ps->cap_params < 8 ? 8 : ps->cap_params * 2;
        slate_tensor_t** new_arr = (slate_tensor_t**)realloc(
            ps->params, (size_t)new_cap * sizeof(*new_arr));
        if (!new_arr) return;
        ps->params = new_arr;
        ps->cap_params = new_cap;
    }
    ps->params[ps->n_params++] = p;
}
