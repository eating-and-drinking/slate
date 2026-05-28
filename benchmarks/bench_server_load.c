// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// bench_server_load.c — measure end-to-end server throughput under
// concurrent load, with and without the micro-batching scheduler.
//
// Spawns one slate inference server on a localhost port, then fires
// N parallel client threads each making M sequential /v1/completions
// calls and measuring tokens/sec.  Runs twice: once with the
// scheduler disabled (cfg.scheduler_max_batch = 0, naive per-request
// decode_step), once with the scheduler enabled (max_batch = 16).

#include "slate/slate.h"
#include "slate/transformer.h"
#include "slate/infer.h"
#include "slate/server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#define PORT      19011
#define VOCAB     4096
#define MAX_SEQ    128
#define D_MODEL    256
#define N_LAYERS     4
#define FFN_H     1024
#define API_KEY  "bench-key"

#define N_CLIENTS    8
#define REQ_PER_CLI  4
#define GEN_TOKENS  16

static double now_sec(void) {
    struct timeval t; gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec / 1e6;
}

// Send one POST /v1/completions request, count returned tokens.
static int hit_completions(int* out_tokens) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(PORT);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) { close(fd); return -1; }

    char body[256];
    int blen = snprintf(body, sizeof(body),
                          "{\"prompt\":[1,2,3,4,5,6,7,8],\"max_tokens\":%d}", GEN_TOKENS);
    char req[1024];
    int rlen = snprintf(req, sizeof(req),
                          "POST /v1/completions HTTP/1.1\r\n"
                          "Host: x\r\n"
                          "Authorization: Bearer %s\r\n"
                          "Content-Type: application/json\r\n"
                          "Content-Length: %d\r\n\r\n%s",
                          API_KEY, blen, body);
    if (send(fd, req, rlen, 0) != rlen) { close(fd); return -1; }

    char buf[8192];
    int total = 0;
    while (total < (int)sizeof(buf) - 1) {
        int got = (int)recv(fd, buf + total, sizeof(buf) - 1 - total, 0);
        if (got <= 0) break;
        total += got;
    }
    buf[total] = '\0';
    close(fd);

    const char* p = strstr(buf, "\"completion_tokens\":");
    if (!p) return -1;
    int got = atoi(p + 20);
    *out_tokens = got;
    return 0;
}

typedef struct {
    int my_id;
    int total_tokens;
    int ok;
} client_arg_t;

static void* client_thread(void* arg) {
    client_arg_t* a = (client_arg_t*)arg;
    a->total_tokens = 0;
    a->ok = 1;
    for (int i = 0; i < REQ_PER_CLI; ++i) {
        int got;
        if (hit_completions(&got) != 0) { a->ok = 0; return NULL; }
        a->total_tokens += got;
    }
    return NULL;
}

static double run_load_test(slate_infer_engine_t* eng, int scheduler_batch_size,
                              const char** label_out) {
    slate_server_config_t cfg = {
        .port = PORT,
        .n_workers = 16,
        .max_tokens = GEN_TOKENS,
        .max_body_kb = 64,
        .shutdown_timeout_sec = 5,
        .scheduler_max_batch = scheduler_batch_size,
        .api_key = API_KEY,
    };
    slate_server_t* srv = slate_server_new(eng, &cfg);
    if (!srv) { *label_out = "server FAIL"; return 0; }
    *label_out = (scheduler_batch_size > 0) ? "with scheduler" : "no scheduler";

    pthread_t srv_t;
    pthread_create(&srv_t, NULL, (void*(*)(void*))slate_server_run, srv);
    usleep(200 * 1000);

    pthread_t client_threads[N_CLIENTS];
    client_arg_t args[N_CLIENTS];
    for (int i = 0; i < N_CLIENTS; ++i) args[i].my_id = i;

    double t0 = now_sec();
    for (int i = 0; i < N_CLIENTS; ++i)
        pthread_create(&client_threads[i], NULL, client_thread, &args[i]);
    for (int i = 0; i < N_CLIENTS; ++i)
        pthread_join(client_threads[i], NULL);
    double t1 = now_sec();

    int total = 0, all_ok = 1;
    for (int i = 0; i < N_CLIENTS; ++i) {
        total += args[i].total_tokens;
        if (!args[i].ok) all_ok = 0;
    }
    double elapsed = t1 - t0;
    double tps = elapsed > 0 ? (double)total / elapsed : 0;

    slate_server_stop(srv);
    pthread_join(srv_t, NULL);
    slate_server_free(srv);

    printf("  %-16s   clients=%d req/cli=%d gen=%d  -> %d tokens in %.3fs  %.1f tok/s  ok=%d\n",
            *label_out, N_CLIENTS, REQ_PER_CLI, GEN_TOKENS,
            total, elapsed, tps, all_ok);
    return tps;
}

int main(void) {
    printf("=== slate server load benchmark ===\n");
    printf("Model: V=%d D=%d L=%d FFN=%d   %d concurrent clients, %d req each, %d tokens each\n\n",
            VOCAB, D_MODEL, N_LAYERS, FFN_H, N_CLIENTS, REQ_PER_CLI, GEN_TOKENS);

    slate_arena_t* P = slate_arena_create(128 << 20);
    slate_module_t* model = slate_module_causal_lm_new(
        P, VOCAB, MAX_SEQ, D_MODEL, N_LAYERS, FFN_H, 1e-5f, /*seed=*/42);
    if (!model) { puts("model build FAIL"); return 1; }

    slate_infer_engine_t* eng = slate_infer_engine_new(
        model, N_LAYERS, D_MODEL, VOCAB, FFN_H, MAX_SEQ);
    if (!eng) { puts("engine new FAIL"); return 1; }

    const char* label;
    double tps_off = run_load_test(eng, /*scheduler_batch_size=*/0,  &label);
    sleep(1);   // let the previous port close cleanly
    double tps_on  = run_load_test(eng, /*scheduler_batch_size=*/16, &label);

    printf("\n=== Speedup: %.2fx ===\n",
            tps_off > 0 ? tps_on / tps_off : 0);

    slate_infer_engine_free(eng);
    slate_module_destroy(model);
    slate_arena_destroy(P);
    return 0;
}
