// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// scheduler.c — single-decoder-thread micro-batcher in front of
// slate_infer_batch_step.  See scheduler.h for the model.

#include "slate/scheduler.h"
#include "slate/infer.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define POOL_CAP 256       // max pending requests in queue

typedef struct sched_req {
    slate_infer_session_t* sess;
    int32_t token;
    float*  logits_out;
    int     vocab;
    int     done;
    int     rc;             // batch_step return code
    pthread_cond_t  cond;
    pthread_mutex_t lock;
} sched_req_t;

struct slate_scheduler {
    slate_infer_engine_t* eng;
    int                   max_batch;
    int                   vocab;
    pthread_mutex_t       pool_lock;
    pthread_cond_t        pool_cond;
    sched_req_t*          pending[POOL_CAP];
    int                   pending_count;
    pthread_t             decoder;
    _Atomic int           running;
    // Stats
    _Atomic uint64_t      total_batches;
    _Atomic uint64_t      total_requests;
};

static void* decoder_main(void* arg) {
    slate_scheduler_t* s = (slate_scheduler_t*)arg;
    slate_infer_batch_t* batch = slate_infer_batch_new(s->eng, s->max_batch);
    int vocab = s->vocab;
    float* all_logits = (float*)malloc((size_t)vocab * s->max_batch * sizeof(float));

    sched_req_t*           drain[POOL_CAP];
    slate_infer_session_t* sess_arr[POOL_CAP];
    int32_t                tok_arr[POOL_CAP];

    while (atomic_load(&s->running)) {
        pthread_mutex_lock(&s->pool_lock);
        while (atomic_load(&s->running) && s->pending_count == 0) {
            pthread_cond_wait(&s->pool_cond, &s->pool_lock);
        }
        if (!atomic_load(&s->running) && s->pending_count == 0) {
            pthread_mutex_unlock(&s->pool_lock);
            break;
        }
        int n = s->pending_count;
        if (n > s->max_batch) n = s->max_batch;
        for (int i = 0; i < n; ++i) {
            drain[i]     = s->pending[i];
            sess_arr[i]  = drain[i]->sess;
            tok_arr[i]   = drain[i]->token;
        }
        // shift remaining (rare path; could also use a ring)
        for (int i = n; i < s->pending_count; ++i) {
            s->pending[i - n] = s->pending[i];
        }
        s->pending_count -= n;
        pthread_mutex_unlock(&s->pool_lock);

        int rc = slate_infer_batch_step(batch, sess_arr, n, tok_arr, all_logits);
        atomic_fetch_add(&s->total_batches, 1);
        atomic_fetch_add(&s->total_requests, (uint64_t)n);

        for (int i = 0; i < n; ++i) {
            sched_req_t* r = drain[i];
            if (rc == 0) {
                memcpy(r->logits_out, all_logits + (size_t)i * vocab,
                        (size_t)vocab * sizeof(float));
            }
            pthread_mutex_lock(&r->lock);
            r->done = 1;
            r->rc   = rc;
            pthread_cond_signal(&r->cond);
            pthread_mutex_unlock(&r->lock);
        }
    }

    free(all_logits);
    slate_infer_batch_free(batch);
    return NULL;
}

slate_scheduler_t* slate_scheduler_new(slate_infer_engine_t* engine,
                                        int max_batch) {
    if (!engine || max_batch <= 0) return NULL;
    if (max_batch > POOL_CAP) max_batch = POOL_CAP;
    slate_scheduler_t* s = (slate_scheduler_t*)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->eng       = engine;
    s->max_batch = max_batch;
    s->vocab     = slate_infer_engine_vocab(engine);
    pthread_mutex_init(&s->pool_lock, NULL);
    pthread_cond_init (&s->pool_cond, NULL);
    atomic_store(&s->running, 1);
    if (pthread_create(&s->decoder, NULL, decoder_main, s) != 0) {
        pthread_mutex_destroy(&s->pool_lock);
        pthread_cond_destroy(&s->pool_cond);
        free(s);
        return NULL;
    }
    return s;
}

void slate_scheduler_free(slate_scheduler_t* s) {
    if (!s) return;
    atomic_store(&s->running, 0);
    pthread_mutex_lock(&s->pool_lock);
    pthread_cond_broadcast(&s->pool_cond);
    pthread_mutex_unlock(&s->pool_lock);
    pthread_join(s->decoder, NULL);
    pthread_mutex_destroy(&s->pool_lock);
    pthread_cond_destroy(&s->pool_cond);
    free(s);
}

int slate_scheduler_decode(slate_scheduler_t* s,
                            slate_infer_session_t* sess,
                            int32_t token, float* out_logits) {
    if (!s || !sess || !out_logits) return -1;
    sched_req_t req;
    memset(&req, 0, sizeof(req));
    req.sess       = sess;
    req.token      = token;
    req.logits_out = out_logits;
    req.vocab      = s->vocab;
    req.done       = 0;
    req.rc         = 0;
    pthread_cond_init (&req.cond, NULL);
    pthread_mutex_init(&req.lock, NULL);

    pthread_mutex_lock(&s->pool_lock);
    if (s->pending_count >= POOL_CAP) {
        pthread_mutex_unlock(&s->pool_lock);
        pthread_cond_destroy(&req.cond);
        pthread_mutex_destroy(&req.lock);
        return -2;   // queue full
    }
    s->pending[s->pending_count++] = &req;
    pthread_cond_signal(&s->pool_cond);
    pthread_mutex_unlock(&s->pool_lock);

    pthread_mutex_lock(&req.lock);
    while (!req.done) pthread_cond_wait(&req.cond, &req.lock);
    int rc = req.rc;
    pthread_mutex_unlock(&req.lock);

    pthread_cond_destroy(&req.cond);
    pthread_mutex_destroy(&req.lock);
    return rc;
}

int slate_scheduler_pending(const slate_scheduler_t* s) {
    if (!s) return 0;
    pthread_mutex_lock((pthread_mutex_t*)&s->pool_lock);
    int n = s->pending_count;
    pthread_mutex_unlock((pthread_mutex_t*)&s->pool_lock);
    return n;
}

uint64_t slate_scheduler_total_batches(const slate_scheduler_t* s) {
    return s ? atomic_load(&s->total_batches) : 0;
}

double slate_scheduler_avg_batch_size(const slate_scheduler_t* s) {
    if (!s) return 0.0;
    uint64_t b = atomic_load(&s->total_batches);
    uint64_t r = atomic_load(&s->total_requests);
    return b > 0 ? (double)r / (double)b : 0.0;
}
