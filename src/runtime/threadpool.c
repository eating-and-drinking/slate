// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// threadpool.c — pthread-backed fixed-size worker pool with atomic task
// dispatch. Used by parallel_for to spread independent task IDs across cores.
//
// Design:
//   - N worker threads created at pool creation, parked on a condvar
//   - parallel_for sets up shared state (fn, user_data, n_tasks, next),
//     signals workers, then waits on a completion barrier
//   - Each worker atomically increments `next` to grab the next task id
//   - When `done_count == active`, main thread unblocks
//
// This is "static round-robin via atomic fetch_add" rather than full
// work-stealing. For uniformly-sized tasks (matmul tiles) it's enough.

#include "slate/runtime.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

struct slate_threadpool {
    int n_threads;
    int active;

    pthread_t* threads;
    pthread_mutex_t mu;
    pthread_cond_t cv_work;
    pthread_cond_t cv_done;

    slate_task_fn fn;
    void* user_data;
    int n_tasks;
    atomic_int next;
    atomic_int done_count;
    int batch_id;          // incremented per batch to wake workers reliably
    int last_seen_batch;   // workers compare; if changed, they have work
    int shutdown;
};

static void* worker_main(void* arg) {
    slate_threadpool_t* p = (slate_threadpool_t*)arg;
    int local_batch = 0;
    while (1) {
        pthread_mutex_lock(&p->mu);
        while (!p->shutdown && p->batch_id == local_batch) {
            pthread_cond_wait(&p->cv_work, &p->mu);
        }
        if (p->shutdown) { pthread_mutex_unlock(&p->mu); break; }
        int my_batch = p->batch_id;
        int my_active = p->active;
        slate_task_fn fn = p->fn;
        void* ud = p->user_data;
        int n_tasks = p->n_tasks;
        pthread_mutex_unlock(&p->mu);

        // Grab tasks atomically until exhausted.
        while (1) {
            int t = atomic_fetch_add(&p->next, 1);
            if (t >= n_tasks) break;
            fn(t, n_tasks, ud);
        }
        local_batch = my_batch;

        pthread_mutex_lock(&p->mu);
        int d = ++(p->done_count);
        if (d == my_active) pthread_cond_signal(&p->cv_done);
        pthread_mutex_unlock(&p->mu);
    }
    return NULL;
}

slate_threadpool_t* slate_threadpool_create(int n_threads) {
    if (n_threads <= 0) n_threads = 1;
    slate_threadpool_t* p = (slate_threadpool_t*)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->n_threads = n_threads;
    p->active = n_threads;
    p->shutdown = 0;
    p->batch_id = 0;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv_work, NULL);
    pthread_cond_init(&p->cv_done, NULL);
    atomic_init(&p->next, 0);
    atomic_init(&p->done_count, 0);

    p->threads = (pthread_t*)calloc((size_t)n_threads, sizeof(pthread_t));
    for (int i = 0; i < n_threads; ++i) {
        pthread_create(&p->threads[i], NULL, worker_main, p);
    }
    return p;
}

void slate_threadpool_destroy(slate_threadpool_t* p) {
    if (!p) return;
    pthread_mutex_lock(&p->mu);
    p->shutdown = 1;
    pthread_cond_broadcast(&p->cv_work);
    pthread_mutex_unlock(&p->mu);
    for (int i = 0; i < p->n_threads; ++i) pthread_join(p->threads[i], NULL);
    free(p->threads);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv_work);
    pthread_cond_destroy(&p->cv_done);
    free(p);
}

int slate_threadpool_num_threads(const slate_threadpool_t* p) {
    return p ? p->n_threads : 0;
}

void slate_threadpool_set_active(slate_threadpool_t* p, int n) {
    if (!p) return;
    if (n < 1) n = 1;
    if (n > p->n_threads) n = p->n_threads;
    pthread_mutex_lock(&p->mu);
    p->active = n;
    pthread_mutex_unlock(&p->mu);
}

void slate_threadpool_parallel_for(slate_threadpool_t* p, int n_tasks,
                                    slate_task_fn fn, void* user_data) {
    if (!fn || n_tasks <= 0) return;
    if (!p || p->n_threads <= 1 || n_tasks == 1) {
        for (int i = 0; i < n_tasks; ++i) fn(i, n_tasks, user_data);
        return;
    }
    pthread_mutex_lock(&p->mu);
    p->fn = fn;
    p->user_data = user_data;
    p->n_tasks = n_tasks;
    atomic_store(&p->next, 0);
    atomic_store(&p->done_count, 0);
    p->batch_id++;
    int wait_for = p->active;
    pthread_cond_broadcast(&p->cv_work);
    while (atomic_load(&p->done_count) < wait_for) {
        pthread_cond_wait(&p->cv_done, &p->mu);
    }
    pthread_mutex_unlock(&p->mu);
}
