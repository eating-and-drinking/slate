// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// pool_global.c — process-wide threadpool used by ops for matmul/bmm/etc.

#include "slate/runtime.h"
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

slate_threadpool_t* slate_global_pool(void);

static slate_threadpool_t* g_pool = NULL;
static pthread_once_t g_init = PTHREAD_ONCE_INIT;

static int detect_n_cpus(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) return 1;
    if (n > 64) return 64;  // sanity cap
    return (int)n;
}

static void init_pool(void) {
    const char* env = getenv("SLATE_NUM_THREADS");
    int n = env ? atoi(env) : detect_n_cpus();
    if (n < 1) n = 1;
    g_pool = slate_threadpool_create(n);
}

slate_threadpool_t* slate_global_pool(void) {
    pthread_once(&g_init, init_pool);
    return g_pool;
}
