// SPDX-License-Identifier: Apache-2.0
#include "slate/runtime.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>

static atomic_int counter;
static void inc_task(int task_id, int n_tasks, void* ud) {
    (void)task_id; (void)n_tasks; (void)ud;
    atomic_fetch_add(&counter, 1);
}

int main(void) {
    slate_threadpool_t* p = slate_threadpool_create(4);
    int n = slate_threadpool_num_threads(p);
    printf("[tp] %d threads\n", n);
    atomic_store(&counter, 0);
    slate_threadpool_parallel_for(p, 10000, inc_task, NULL);
    int c = atomic_load(&counter);
    printf("[tp] expected 10000, got %d\n", c);
    int ok = (c == 10000);

    // Multiple batches
    for (int i = 0; i < 5; ++i) {
        atomic_store(&counter, 0);
        slate_threadpool_parallel_for(p, 1000, inc_task, NULL);
        if (atomic_load(&counter) != 1000) ok = 0;
    }
    slate_threadpool_destroy(p);
    printf("%s\n", ok ? "test_threadpool: OK" : "test_threadpool: FAIL");
    return ok ? 0 : 1;
}
