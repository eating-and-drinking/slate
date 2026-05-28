// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// 11_inference_server — end-to-end inference server demo.
//
// Demonstrates:
//   * slate_module_causal_lm_new + slate_infer_engine_new
//   * slate_apikey_set_t with two keys (one with rate limit, one without)
//   * slate_server_new + slate_server_run on a background thread
//   * a built-in HTTP client that hits:
//       /health
//       /v1/completions (no auth, key A, key B, with stream:true)
//       /v1/completions (key A again to trigger 429 rate-limit)
//       /metrics
//   * graceful shutdown with slate_server_install_signal_handler
//
// All exits 0 on full pass.

#include "slate/slate.h"
#include "slate/transformer.h"
#include "slate/infer.h"
#include "slate/server.h"
#include "slate/apikey.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define PORT        18801
#define VOCAB          32
#define MAX_SEQ        32
#define D_MODEL        32
#define N_LAYERS        2
#define FFN_H          64
#define KEY_HEAVY  "key-heavy"   // rate-limited
#define KEY_LIGHT  "key-light"   // unlimited

typedef struct { slate_server_t* srv; } thread_args_t;
static void* server_thread(void* a) {
    slate_server_run(((thread_args_t*)a)->srv);
    return NULL;
}

static int http_call(const char* request, char* out, int out_cap, int* out_status) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(PORT);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) { close(fd); return -1; }
    int n = (int)strlen(request);
    if (send(fd, request, n, 0) != n) { close(fd); return -1; }
    int total = 0;
    while (total < out_cap - 1) {
        int got = (int)recv(fd, out + total, out_cap - 1 - total, 0);
        if (got <= 0) break;
        total += got;
    }
    out[total] = '\0';
    close(fd);
    int st = 0;
    sscanf(out, "HTTP/1.1 %d", &st);
    if (out_status) *out_status = st;
    return total;
}

// Build a POST /v1/completions request with the given body + Bearer key.
static int build_completion_req(char* req, int cap,
                                  const char* api_key,
                                  const char* body) {
    return snprintf(req, cap,
                     "POST /v1/completions HTTP/1.1\r\n"
                     "Host: x\r\n"
                     "Authorization: Bearer %s\r\n"
                     "Content-Type: application/json\r\n"
                     "Content-Length: %zu\r\n\r\n%s",
                     api_key, strlen(body), body);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== slate inference server demo ===\n");

    // 1. Build a tiny transformer.
    slate_arena_t* P = slate_arena_create(8 << 20);
    slate_module_t* model = slate_module_causal_lm_new(
        P, VOCAB, MAX_SEQ, D_MODEL, N_LAYERS, FFN_H, 1e-5f, /*seed=*/0xABCD);
    if (!model) { puts("model build FAIL"); return 1; }

    slate_infer_engine_t* eng = slate_infer_engine_new(
        model, N_LAYERS, D_MODEL, VOCAB, FFN_H, MAX_SEQ);
    if (!eng) { puts("engine new FAIL"); return 1; }

    // 2. Multi-key setup: KEY_HEAVY has rps=2/burst=2, KEY_LIGHT is unlimited.
    slate_apikey_set_t* keys = slate_apikey_set_new();
    slate_apikey_set_add(keys, KEY_HEAVY, "heavy-tier", /*rps=*/2.0, /*burst=*/2);
    slate_apikey_set_add(keys, KEY_LIGHT, "light-tier", /*rps=*/0.0, /*burst=*/0);

    // 3. Launch the server.
    slate_server_config_t cfg = {
        .port = PORT,
        .n_workers = 4,
        .max_tokens = 8,
        .max_body_kb = 64,
        .shutdown_timeout_sec = 5,
        .apikey_set = keys,
    };
    slate_server_t* srv = slate_server_new(eng, &cfg);
    if (!srv) { puts("server new FAIL (port busy?)"); return 1; }
    slate_server_install_signal_handler(srv);
    printf("server bound on 127.0.0.1:%d   keys: heavy(2 rps), light(unlimited)\n", PORT);

    pthread_t st;
    thread_args_t ta = { .srv = srv };
    pthread_create(&st, NULL, server_thread, &ta);
    usleep(150 * 1000);

    int ok = 1;
    char resp[16384];

    // -- /health
    {
        int st_code = 0;
        http_call("GET /health HTTP/1.1\r\nHost: x\r\n\r\n", resp, sizeof(resp), &st_code);
        printf("[1] /health                 -> %d\n", st_code);
        if (st_code != 200) ok = 0;
    }

    // -- no auth -> 401
    {
        char req[512];
        snprintf(req, sizeof(req),
                  "POST /v1/completions HTTP/1.1\r\nHost: x\r\n"
                  "Content-Type: application/json\r\nContent-Length: 22\r\n\r\n"
                  "{\"prompt\":[1,2,3,4,5]}");
        int st_code = 0;
        http_call(req, resp, sizeof(resp), &st_code);
        printf("[2] /v1/completions no-auth -> %d\n", st_code);
        if (st_code != 401) ok = 0;
    }

    // -- key-light, normal response
    {
        char req[1024];
        build_completion_req(req, sizeof(req), KEY_LIGHT,
                              "{\"prompt\":[1,2,3],\"max_tokens\":4}");
        int st_code = 0;
        http_call(req, resp, sizeof(resp), &st_code);
        const char* body_p = strstr(resp, "\r\n\r\n");
        printf("[3] light key, max=4 toks    -> %d\n  body: %.150s\n",
                st_code, body_p ? body_p + 4 : resp);
        if (st_code != 200) ok = 0;
        if (!strstr(resp, "\"key_label\":\"light-tier\"")) ok = 0;
    }

    // -- key-light, streaming
    {
        char req[1024];
        build_completion_req(req, sizeof(req), KEY_LIGHT,
                              "{\"prompt\":[1,2,3],\"max_tokens\":3,\"stream\":true}");
        int st_code = 0;
        http_call(req, resp, sizeof(resp), &st_code);
        printf("[4] light key, stream=true   -> %d\n", st_code);
        // Look for SSE markers
        int seen_data = 0, seen_done = 0;
        if (strstr(resp, "text/event-stream")) seen_data++;
        const char* p = resp;
        while ((p = strstr(p, "data: {\"token\":")) != NULL) { seen_data++; p++; }
        if (strstr(resp, "data: [DONE]")) seen_done = 1;
        printf("  SSE markers: data-events=%d, [DONE]=%d\n", seen_data, seen_done);
        if (st_code != 200 || seen_data < 2 || !seen_done) ok = 0;
    }

    // -- key-heavy: burst is 2, so two requests succeed, third gets 429.
    {
        char req[1024];
        build_completion_req(req, sizeof(req), KEY_HEAVY,
                              "{\"prompt\":[1,2,3],\"max_tokens\":1}");
        int codes[3];
        for (int i = 0; i < 3; ++i) {
            int st_code = 0;
            http_call(req, resp, sizeof(resp), &st_code);
            codes[i] = st_code;
        }
        printf("[5] heavy key x3 (burst=2)   -> %d, %d, %d\n",
                codes[0], codes[1], codes[2]);
        if (codes[0] != 200 || codes[1] != 200 || codes[2] != 429) ok = 0;
    }

    // -- /metrics
    {
        int st_code = 0;
        http_call("GET /metrics HTTP/1.1\r\nHost: x\r\n\r\n", resp, sizeof(resp), &st_code);
        printf("[6] /metrics                -> %d\n", st_code);
        if (st_code != 200) ok = 0;
        const char* required[] = {
            "slate_requests_total",
            "slate_rate_limited_total",
            "slate_stream_requests_total",
            "slate_auth_failures_total",
            "slate_time_to_first_token_ms",
            NULL
        };
        for (int i = 0; required[i]; ++i) {
            if (!strstr(resp, required[i])) {
                printf("  MISSING metric: %s\n", required[i]);
                ok = 0;
            }
        }
        // Print a few key counters
        const char* sub[] = {
            "slate_requests_total ",
            "slate_rate_limited_total ",
            "slate_stream_requests_total ",
            "slate_auth_failures_total ",
            NULL
        };
        for (int i = 0; sub[i]; ++i) {
            const char* h = strstr(resp, sub[i]);
            if (h) {
                const char* nl = strchr(h, '\n');
                if (nl) printf("  %.*s\n", (int)(nl - h), h);
            }
        }
    }

    // -- After graceful-stop signal, /health returns 503.  We can't
    //    easily raise the signal mid-test, but slate_server_stop has the
    //    same effect — simulate by calling it directly, then probe.
    slate_server_stop(srv);
    usleep(50 * 1000);
    {
        int st_code = 0;
        http_call("GET /health HTTP/1.1\r\nHost: x\r\n\r\n", resp, sizeof(resp), &st_code);
        printf("[7] /health (draining)      -> %d\n", st_code);
        // Either 503 (handled before close) or connection refused (after close).
        if (st_code != 503 && st_code != 0) ok = 0;
    }

    pthread_join(st, NULL);
    slate_server_free(srv);
    slate_apikey_set_free(keys);
    slate_infer_engine_free(eng);
    slate_module_destroy(model);
    slate_arena_destroy(P);

    printf("\n=== %s ===\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
