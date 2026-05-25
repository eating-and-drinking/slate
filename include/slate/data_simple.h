// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// data_simple.h — synchronous in-memory data utilities (M1).
//
// The full async streaming DataLoader is in data.h and arrives in M5. This
// header carries the simpler primitives needed for MNIST-class problems:
//
//   - IDX file parsing (the MNIST native format)
//   - In-memory index shuffler that yields batch indices

#ifndef SLATE_DATA_SIMPLE_H
#define SLATE_DATA_SIMPLE_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// IDX file format (MNIST and friends).
// =============================================================================
//
// Layout (big-endian):
//   magic[4]     : 0x00000801 (uint8, 1D) for labels, 0x00000803 (uint8, 3D) for images
//   n_items[4]   : number of items
//   [for images] rows[4], cols[4]
//   data[n_items * rows * cols] : raw uint8 pixels, or labels

typedef struct slate_idx_data {
    int     n_items;
    int     rows;     // 1 for label files
    int     cols;     // 1 for label files
    uint8_t* data;    // malloc'd; owns the buffer
} slate_idx_data_t;

slate_status_t slate_idx_load_images(const char* path, slate_idx_data_t* out);
slate_status_t slate_idx_load_labels(const char* path, slate_idx_data_t* out);
void slate_idx_free(slate_idx_data_t* d);

// =============================================================================
// SimpleDataloader: emits shuffled batch indices over a finite dataset.
// =============================================================================
//
// Usage:
//   slate_simple_dataloader_t* dl = slate_simple_dataloader_new(60000, 64, true, 42);
//   int idx[64];
//   while (slate_simple_dataloader_next(dl, idx)) {
//       // gather samples by idx[0..63] from your in-memory dataset
//   }
//   slate_simple_dataloader_reset(dl);   // start a new epoch

typedef struct slate_simple_dataloader slate_simple_dataloader_t;

slate_simple_dataloader_t* slate_simple_dataloader_new(int n_samples,
                                                        int batch_size,
                                                        bool shuffle,
                                                        uint64_t seed);

// Fills `indices_out` (length = batch_size) with the next batch's indices.
// Returns false at end of epoch. The last batch may be partial; in that case
// it is dropped (drop_last semantics).
bool slate_simple_dataloader_next(slate_simple_dataloader_t* dl, int* indices_out);

void slate_simple_dataloader_reset(slate_simple_dataloader_t* dl);
int  slate_simple_dataloader_n_batches(const slate_simple_dataloader_t* dl);
void slate_simple_dataloader_destroy(slate_simple_dataloader_t* dl);

#ifdef __cplusplus
}
#endif

#endif // SLATE_DATA_SIMPLE_H
