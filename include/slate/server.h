// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// server.h — production-ready HTTP/1.1 inference server.
//
// Endpoints:
//   POST /v1/completions     JSON body:
//                              { "prompt":[...], "max_tokens":N,
//                                "temperature":1.0, "top_p":0.9,
//                                "stream": false }
//                            Non-streaming response:
//                              { "tokens":[...], "prompt_tokens":N,
//                                "completion_tokens":M, "latency_ms":...,
//                                "key_label":"..." }
//                            Streaming response (when "stream": true):
//                              Content-Type: text/event-stream
//                              data: {"token":N}\n\n        (one per token)
//                              data: [DONE]\n\n             (terminator)
//   GET  /health             "ok\n"
//   GET  /metrics            Prometheus exposition format
//
// Auth:
//   * If `apikey_set` is non-NULL, requires `Authorization: Bearer <k>`;
//     each registered key has its own token-bucket rate limit and emits
//     metrics labelled with the key's `label`.
//   * Else if `api_key` is set, falls back to single-key auth with no
//     rate limit (back-compat with the M5 demo).
//   * Else auth is disabled (development mode).
//
// Concurrency:
//   * One worker thread per connection from a fixed-size pool.
//   * Each worker owns its slate_infer_session_t — KV caches do NOT
//     cross requests.
//
// Graceful shutdown:
//   * slate_server_install_signal_handler() wires SIGINT/SIGTERM to
//     slate_server_stop().
//   * slate_server_free() waits up to shutdown_timeout_sec for
//     in-flight requests to drain before joining workers.

#ifndef SLATE_SERVER_H
#define SLATE_SERVER_H

#include "slate/infer.h"
#include "slate/apikey.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_server slate_server_t;

typedef struct {
    int                  port;
    int                  n_workers;
    int                  max_tokens;
    int                  max_body_kb;
    int                  shutdown_timeout_sec;   // default 30
    int                  scheduler_max_batch;    // 0 = disable batching, default 16
    const char*          api_key;                // optional single-key fallback
    slate_apikey_set_t*  apikey_set;             // preferred multi-key + rate-limit
} slate_server_config_t;

slate_server_t* slate_server_new(slate_infer_engine_t* engine,
                                  const slate_server_config_t* cfg);

// Run the accept loop in this thread until slate_server_stop is
// called (typically from a signal handler).
int slate_server_run(slate_server_t* srv);

// Mark the server as shutting down: accept loop exits, workers drain
// then terminate.  Safe to call from a signal handler — only uses
// async-signal-safe operations (atomic store + shutdown(2)).
void slate_server_stop(slate_server_t* srv);

// Install SIGINT/SIGTERM handlers that call slate_server_stop on the
// most-recently-installed server.  Use for the common case where a
// single server runs in the process.
void slate_server_install_signal_handler(slate_server_t* srv);

void slate_server_free(slate_server_t* srv);

#ifdef __cplusplus
}
#endif

#endif // SLATE_SERVER_H
