// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// mmap_dataset.h — memory-mapped packed-tokens dataset.
//
// The on-disk format is the simplest possible: a single int32 stream
// (little-endian) of tokens. Open with slate_mmap_open(path), sample
// batches via slate_mmap_sample_batch().

#ifndef SLATE_MMAP_DATASET_H
#define SLATE_MMAP_DATASET_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_mmap_dataset slate_mmap_dataset_t;

slate_mmap_dataset_t* slate_mmap_open(const char* path);
void slate_mmap_close(slate_mmap_dataset_t* d);

int64_t slate_mmap_n_tokens(const slate_mmap_dataset_t* d);

// Sample `batch_size` random sequences each of length `seq_len + 1` from
// the dataset. Writes inputs[batch*seq_len] and targets[batch*seq_len]
// (next-token-prediction shifted by one). Returns 0 on success.
int slate_mmap_sample_batch(slate_mmap_dataset_t* d,
                             int batch_size, int seq_len,
                             int32_t* inputs, int32_t* targets,
                             uint64_t* rng_state);

#ifdef __cplusplus
}
#endif

#endif
