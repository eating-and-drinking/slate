// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// metrics.h — Prometheus-style metrics for the inference server.
//
// Three metric types: counters (monotonic), gauges (current value),
// histograms (latency / size distributions). All thread-safe via atomic
// adds; readers (the /metrics endpoint scrape) see a consistent
// snapshot since each counter is read with __atomic_load.
//
// Registry is global and process-wide — same as Prometheus's model.
// Construct metrics with slate_metrics_*_new(name, help) once at
// startup; access them by name in hot paths.

#ifndef SLATE_METRICS_H
#define SLATE_METRICS_H

#include "slate/types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_counter   slate_counter_t;
typedef struct slate_gauge     slate_gauge_t;
typedef struct slate_histogram slate_histogram_t;

// One-shot registry init; call before any metric is created. Safe to
// call multiple times (idempotent).
void slate_metrics_init(void);

// Counter: monotonically increasing 64-bit integer (we promote to
// float in exposition for compatibility).
slate_counter_t* slate_counter_new(const char* name, const char* help);
void             slate_counter_inc(slate_counter_t* c);
void             slate_counter_add(slate_counter_t* c, uint64_t n);
uint64_t         slate_counter_get(const slate_counter_t* c);

// Gauge: 64-bit signed integer that can go up or down.
slate_gauge_t* slate_gauge_new(const char* name, const char* help);
void           slate_gauge_set(slate_gauge_t* g, int64_t v);
void           slate_gauge_add(slate_gauge_t* g, int64_t delta);
int64_t        slate_gauge_get(const slate_gauge_t* g);

// Histogram: fixed-bucket boundaries. Observes values in milliseconds.
// Default boundaries cover 1ms..32s in powers of 2 (suitable for
// inference latency).
slate_histogram_t* slate_histogram_new(const char* name, const char* help);
void               slate_histogram_observe(slate_histogram_t* h, double value);

// Write all registered metrics in Prometheus exposition format to the
// caller's buffer. Returns the number of bytes written (excluding
// terminator) or -1 if the buffer was too small.
int slate_metrics_render(char* buf, size_t cap);

// Number of registered metrics across all types.
int slate_metrics_count(void);

#ifdef __cplusplus
}
#endif

#endif // SLATE_METRICS_H
