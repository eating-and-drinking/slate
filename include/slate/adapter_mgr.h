// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// adapter_mgr.h — file-system lifecycle for LoRA adapters.
//
// Layout under root_dir:
//   current.lora            ← active adapter (used by inference)
//   training.lora.tmp       ← in-progress training adapter
//   archive/YYYY-MM-DD-HHMMSS.lora
//
// All transitions are atomic: write tmp → fsync → rename to target.

#ifndef SLATE_ADAPTER_MGR_H
#define SLATE_ADAPTER_MGR_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_adapter_mgr slate_adapter_mgr_t;

slate_adapter_mgr_t* slate_adapter_mgr_open(const char* root_dir);
void slate_adapter_mgr_close(slate_adapter_mgr_t* m);

// Write candidate to `training.lora.tmp` (atomic via tmp+rename).
slate_status_t slate_adapter_mgr_write_candidate(slate_adapter_mgr_t* m,
                                                  const void* data, size_t size);

// Promote candidate to current. Atomically archive old current first.
slate_status_t slate_adapter_mgr_promote(slate_adapter_mgr_t* m);

// Roll back to a specific archived adapter (by file name in archive/).
slate_status_t slate_adapter_mgr_rollback(slate_adapter_mgr_t* m, const char* archive_name);

// Read the active adapter into a malloc'd buffer.
slate_status_t slate_adapter_mgr_read_current(slate_adapter_mgr_t* m,
                                               void** out_data, size_t* out_size);

// How many adapters are archived.
int slate_adapter_mgr_archive_count(slate_adapter_mgr_t* m);

#ifdef __cplusplus
}
#endif

#endif
