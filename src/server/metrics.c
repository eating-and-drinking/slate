// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// metrics.c — Prometheus-style counters/gauges/histograms.  Atomic
// adds for thread safety; lock the registry only at construction time.
// Exposition writes are best-effort consistent (each counter is
// __atomic_load'd individually).

#include "slate/metrics.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUCKET_COUNT 14
// Bucket upper bounds in milliseconds. Standard "powers of 2 from 1ms"
// shape that captures both fast prompt-only requests and long
// generations.
static const double k_bucket_bounds_ms[BUCKET_COUNT] = {
    1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0,
    256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0
};

typedef enum { METRIC_COUNTER, METRIC_GAUGE, METRIC_HISTOGRAM } metric_kind_t;

struct slate_counter {
    metric_kind_t kind;
    const char* name;
    const char* help;
    uint64_t value;        // monotonic
};

struct slate_gauge {
    metric_kind_t kind;
    const char* name;
    const char* help;
    int64_t value;
};

struct slate_histogram {
    metric_kind_t kind;
    const char* name;
    const char* help;
    uint64_t buckets[BUCKET_COUNT];   // cumulative-le counts
    uint64_t count_inf;               // +Inf bucket count == total observations
    double   sum_value;               // running sum (for _sum metric)
    pthread_mutex_t lock;             // for sum_value (atomic on int but not double)
};

#define MAX_METRICS 256
static void* g_metrics[MAX_METRICS];
static int   g_n_metrics = 0;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static int   g_inited = 0;

void slate_metrics_init(void) {
    pthread_mutex_lock(&g_lock);
    g_inited = 1;
    pthread_mutex_unlock(&g_lock);
}

static void register_metric(void* m) {
    pthread_mutex_lock(&g_lock);
    if (g_n_metrics < MAX_METRICS) g_metrics[g_n_metrics++] = m;
    pthread_mutex_unlock(&g_lock);
}

int slate_metrics_count(void) {
    pthread_mutex_lock(&g_lock);
    int n = g_n_metrics;
    pthread_mutex_unlock(&g_lock);
    return n;
}

// ---------------------------------------------------------------------------
// Counter
// ---------------------------------------------------------------------------
slate_counter_t* slate_counter_new(const char* name, const char* help) {
    slate_counter_t* c = (slate_counter_t*)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->kind = METRIC_COUNTER;
    c->name = name;
    c->help = help;
    c->value = 0;
    register_metric(c);
    return c;
}
void slate_counter_inc(slate_counter_t* c) {
    if (!c) return;
    __atomic_add_fetch(&c->value, 1, __ATOMIC_RELAXED);
}
void slate_counter_add(slate_counter_t* c, uint64_t n) {
    if (!c) return;
    __atomic_add_fetch(&c->value, n, __ATOMIC_RELAXED);
}
uint64_t slate_counter_get(const slate_counter_t* c) {
    if (!c) return 0;
    return __atomic_load_n((uint64_t*)&c->value, __ATOMIC_RELAXED);
}

// ---------------------------------------------------------------------------
// Gauge
// ---------------------------------------------------------------------------
slate_gauge_t* slate_gauge_new(const char* name, const char* help) {
    slate_gauge_t* g = (slate_gauge_t*)calloc(1, sizeof(*g));
    if (!g) return NULL;
    g->kind = METRIC_GAUGE;
    g->name = name;
    g->help = help;
    register_metric(g);
    return g;
}
void slate_gauge_set(slate_gauge_t* g, int64_t v) {
    if (!g) return;
    __atomic_store_n(&g->value, v, __ATOMIC_RELAXED);
}
void slate_gauge_add(slate_gauge_t* g, int64_t delta) {
    if (!g) return;
    __atomic_add_fetch(&g->value, delta, __ATOMIC_RELAXED);
}
int64_t slate_gauge_get(const slate_gauge_t* g) {
    if (!g) return 0;
    return __atomic_load_n((int64_t*)&g->value, __ATOMIC_RELAXED);
}

// ---------------------------------------------------------------------------
// Histogram
// ---------------------------------------------------------------------------
slate_histogram_t* slate_histogram_new(const char* name, const char* help) {
    slate_histogram_t* h = (slate_histogram_t*)calloc(1, sizeof(*h));
    if (!h) return NULL;
    h->kind = METRIC_HISTOGRAM;
    h->name = name;
    h->help = help;
    pthread_mutex_init(&h->lock, NULL);
    register_metric(h);
    return h;
}
void slate_histogram_observe(slate_histogram_t* h, double value) {
    if (!h) return;
    pthread_mutex_lock(&h->lock);
    h->sum_value += value;
    h->count_inf++;
    for (int i = 0; i < BUCKET_COUNT; ++i) {
        if (value <= k_bucket_bounds_ms[i]) {
            h->buckets[i]++;
        }
    }
    pthread_mutex_unlock(&h->lock);
}

// ---------------------------------------------------------------------------
// Exposition
// ---------------------------------------------------------------------------
static int append_str(char* buf, size_t cap, int pos, const char* s) {
    size_t l = strlen(s);
    if ((size_t)pos + l >= cap) return -1;
    memcpy(buf + pos, s, l);
    return pos + (int)l;
}

int slate_metrics_render(char* buf, size_t cap) {
    if (!buf || cap == 0) return -1;
    int pos = 0;
    char line[512];

    pthread_mutex_lock(&g_lock);
    int n = g_n_metrics;
    void* arr[MAX_METRICS];
    for (int i = 0; i < n; ++i) arr[i] = g_metrics[i];
    pthread_mutex_unlock(&g_lock);

    for (int i = 0; i < n; ++i) {
        metric_kind_t kind = *(metric_kind_t*)arr[i];
        if (kind == METRIC_COUNTER) {
            slate_counter_t* c = (slate_counter_t*)arr[i];
            snprintf(line, sizeof(line), "# HELP %s %s\n# TYPE %s counter\n%s %llu\n",
                      c->name, c->help, c->name, c->name,
                      (unsigned long long)slate_counter_get(c));
            int n2 = append_str(buf, cap, pos, line);
            if (n2 < 0) return -1; pos = n2;
        } else if (kind == METRIC_GAUGE) {
            slate_gauge_t* g = (slate_gauge_t*)arr[i];
            snprintf(line, sizeof(line), "# HELP %s %s\n# TYPE %s gauge\n%s %lld\n",
                      g->name, g->help, g->name, g->name,
                      (long long)slate_gauge_get(g));
            int n2 = append_str(buf, cap, pos, line);
            if (n2 < 0) return -1; pos = n2;
        } else if (kind == METRIC_HISTOGRAM) {
            slate_histogram_t* h = (slate_histogram_t*)arr[i];
            pthread_mutex_lock(&h->lock);
            uint64_t buckets[BUCKET_COUNT];
            uint64_t count_inf;
            double   sum_value;
            memcpy(buckets, h->buckets, sizeof(buckets));
            count_inf  = h->count_inf;
            sum_value  = h->sum_value;
            pthread_mutex_unlock(&h->lock);
            snprintf(line, sizeof(line), "# HELP %s %s\n# TYPE %s histogram\n",
                      h->name, h->help, h->name);
            int n2 = append_str(buf, cap, pos, line);
            if (n2 < 0) return -1; pos = n2;
            for (int b = 0; b < BUCKET_COUNT; ++b) {
                snprintf(line, sizeof(line), "%s_bucket{le=\"%.0f\"} %llu\n",
                          h->name, k_bucket_bounds_ms[b], (unsigned long long)buckets[b]);
                n2 = append_str(buf, cap, pos, line);
                if (n2 < 0) return -1; pos = n2;
            }
            snprintf(line, sizeof(line),
                      "%s_bucket{le=\"+Inf\"} %llu\n%s_sum %.6f\n%s_count %llu\n",
                      h->name, (unsigned long long)count_inf,
                      h->name, sum_value,
                      h->name, (unsigned long long)count_inf);
            n2 = append_str(buf, cap, pos, line);
            if (n2 < 0) return -1; pos = n2;
        }
    }
    if ((size_t)pos < cap) buf[pos] = '\0';
    return pos;
}
