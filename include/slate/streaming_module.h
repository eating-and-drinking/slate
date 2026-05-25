// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// streaming_module.h — modules whose weights live on disk and are loaded
// only when forward() is in flight.

#ifndef SLATE_STREAMING_MODULE_H
#define SLATE_STREAMING_MODULE_H

#include "slate/types.h"
#include "slate/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Streaming linear: y = x @ W   (no bias). Weight stored at `weights_path`.
// Forward mmaps + computes + munmaps. The arena is used only for the module
// metadata; no permanent RAM is held for the weights.
slate_module_t* slate_module_streaming_linear_new(const char* weights_path,
                                                   int in_features,
                                                   int out_features);

// Diagnostic: peak resident bytes seen by any streaming module since startup.
size_t slate_streaming_peak_bytes(void);
void slate_streaming_reset_peak(void);

#ifdef __cplusplus
}
#endif

#endif
