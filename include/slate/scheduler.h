// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// scheduler.h — micro-batching decode pool that wraps slate_infer_batch.
//
// The HTTP server has N worker threads, one per active connection, each
// performing prefill independently (cheap; each session is independent)
// and then a per-token decode loop.  Naively each worker calls
// slate_infer_decode_step which dispatches its linear projections as
// M=1 GEMMs — wasting the GEMM kernel's tile cache.  The scheduler is
// a single dedicated decoder thread that:
//
//   * holds a slate_infer_batch_t of capacity max_batch;
//   * pulls up to max_batch pending requests at each iteration;
//   * runs ONE M=B GEMM per layer for those requests, then distributes
//     the logits back to each waiting worker.
//
// Workers call slate_scheduler_decode(...) which is synchronous: it
// pushes the request into the queue, blocks on a per-request cond var,
// and returns when the decoder thread has filled `out_logits`.  Workers
// can leave between tokens (max_tokens reached, EOS hit, etc.) without
// breaking the batch — the scheduler only sees one request at a time
// per worker.

#ifndef SLATE_SCHEDULER_H
#define SLATE_SCHEDULER_H

#include "slate/infer.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_scheduler slate_scheduler_t;

// Create a scheduler that batches up to `max_batch` requests per
// iteration and starts one dedicated decoder thread.  The engine
// must outlive the scheduler.
slate_scheduler_t* slate_scheduler_new(slate_infer_engine_t* engine,
                                        int max_batch);

// Stops the decoder thread (signals exit, joins) and frees resources.
void slate_scheduler_free(slate_scheduler_t* s);

// Block until the decoder thread has processed this token for `sess`,
// writing `vocab` floats into `out_logits`.  Returns 0 on success,
// < 0 on error (engine mismatch, out-of-range token, etc.).
//
// Safe to call from many threads concurrently — each call is queued
// independently and batched with others that arrive concurrently.
int slate_scheduler_decode(slate_scheduler_t* s,
                            slate_infer_session_t* sess,
                            int32_t token,
                            float* out_logits);

// Snapshot stats (for /metrics).
int slate_scheduler_pending(const slate_scheduler_t* s);
uint64_t slate_scheduler_total_batches(const slate_scheduler_t* s);
double  slate_scheduler_avg_batch_size(const slate_scheduler_t* s);

#ifdef __cplusplus
}
#endif

#endif // SLATE_SCHEDULER_H
