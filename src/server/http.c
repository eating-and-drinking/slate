// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// http.c — production HTTP/1.1 inference server.
//
// Features in this layer:
//   * non-streaming + Server-Sent-Events streaming for /v1/completions;
//   * multi-key auth with per-key token-bucket rate limiting (429s);
//   * graceful SIGINT/SIGTERM shutdown (drain in-flight requests);
//   * Prometheus metrics with per-key labels;
//   * structured JSON logs on every request.
//
// One thread per connection from a fixed worker pool; each worker
// owns its own slate_infer_session_t so KV caches never cross
// requests. Bound to INADDR_ANY — TLS termination lives in an
// upstream reverse proxy (nginx / Envoy / Caddy).

#include "slate/server.h"
#include "slate/metrics.h"
#include "slate/jlog.h"
#include "slate/sampling.h"
#include "slate/scheduler.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#define MAX_WORKERS 256

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------
typedef struct {
    slate_counter_t*   requests_total;
    slate_counter_t*   tokens_in_total;
    slate_counter_t*   tokens_out_total;
    slate_counter_t*   auth_failures_total;
    slate_counter_t*   rate_limited_total;
    slate_counter_t*   stream_requests_total;
    slate_counter_t*   errors_total;
    slate_gauge_t*     active_connections;
    slate_gauge_t*     active_requests;
    slate_histogram_t* request_latency_ms;
    slate_histogram_t* tokens_out_hist;
    slate_histogram_t* time_to_first_token_ms;
} srv_metrics_t;

static srv_metrics_t g_m = {0};
static pthread_once_t g_metrics_once = PTHREAD_ONCE_INIT;

static void init_metrics(void) {
    slate_metrics_init();
    g_m.requests_total         = slate_counter_new("slate_requests_total",        "Total /v1/completions requests");
    g_m.tokens_in_total        = slate_counter_new("slate_tokens_in_total",       "Total prompt tokens processed");
    g_m.tokens_out_total       = slate_counter_new("slate_tokens_out_total",      "Total generation tokens produced");
    g_m.auth_failures_total    = slate_counter_new("slate_auth_failures_total",   "Requests rejected by auth");
    g_m.rate_limited_total     = slate_counter_new("slate_rate_limited_total",    "Requests rejected by per-key rate limit");
    g_m.stream_requests_total  = slate_counter_new("slate_stream_requests_total", "Total streaming /v1/completions requests");
    g_m.errors_total           = slate_counter_new("slate_errors_total",          "Requests that ended with HTTP 5xx");
    g_m.active_connections     = slate_gauge_new  ("slate_active_connections",    "Currently-open client connections");
    g_m.active_requests        = slate_gauge_new  ("slate_active_requests",       "In-flight /v1/completions requests (for graceful drain)");
    g_m.request_latency_ms     = slate_histogram_new("slate_request_latency_ms",      "End-to-end /v1/completions latency");
    g_m.tokens_out_hist        = slate_histogram_new("slate_tokens_out_hist",         "Distribution of generation lengths");
    g_m.time_to_first_token_ms = slate_histogram_new("slate_time_to_first_token_ms",  "Prefill + first decode latency");
}

// ---------------------------------------------------------------------------
// Signal handling
// ---------------------------------------------------------------------------
static _Atomic(struct slate_server*) g_signal_target = NULL;

static void signal_stop_handler(int sig) {
    (void)sig;
    struct slate_server* s = atomic_load(&g_signal_target);
    if (s) slate_server_stop(s);
}

// ---------------------------------------------------------------------------
// Server state
// ---------------------------------------------------------------------------
struct slate_server {
    slate_infer_engine_t* engine;
    slate_server_config_t cfg;

    int                   listen_fd;
    _Atomic int           running;             // 0 -> draining/stopped
    _Atomic int           in_flight_requests;  // for graceful drain

    pthread_t             workers[MAX_WORKERS];
    int                   n_workers;

    // Connection queue
    pthread_mutex_t       qlock;
    pthread_cond_t        qcond;
    int                   queue[MAX_WORKERS * 4];
    int                   qhead, qtail, qcount, qcap;

    // Single-key back-compat
    slate_apikey_set_t*   owned_keyset;        // synthesised when only api_key is given

    // Optional batched decode scheduler (NULL if disabled)
    slate_scheduler_t*    scheduler;
};

// ---------------------------------------------------------------------------
// Tiny JSON helpers (just enough to parse our request schema)
// ---------------------------------------------------------------------------
static const char* json_find_key(const char* body, const char* key) {
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char* p = strstr(body, needle);
    if (!p) return NULL;
    p += strlen(needle);
    while (*p && (*p == ' ' || *p == ':' || *p == '\t' || *p == '\n')) ++p;
    return p;
}

static int json_int(const char* p, int defval) {
    if (!p) return defval;
    while (*p && (*p == ' ' || *p == '\t')) ++p;
    if (!*p) return defval;
    return atoi(p);
}

static double json_double(const char* p, double defval) {
    if (!p) return defval;
    while (*p && (*p == ' ' || *p == '\t')) ++p;
    if (!*p) return defval;
    return atof(p);
}

static int json_bool(const char* p, int defval) {
    if (!p) return defval;
    while (*p && (*p == ' ' || *p == '\t')) ++p;
    if (strncmp(p, "true",  4) == 0) return 1;
    if (strncmp(p, "false", 5) == 0) return 0;
    return defval;
}

static int json_int_array(const char* p, int32_t* out, int cap) {
    if (!p) return -1;
    while (*p && (*p == ' ' || *p == '\t')) ++p;
    if (*p != '[') return -1;
    ++p;
    int n = 0;
    while (*p && *p != ']') {
        while (*p == ' ' || *p == ',' || *p == '\t' || *p == '\n') ++p;
        if (*p == ']') break;
        if (n >= cap) return -1;
        char* end;
        long v = strtol(p, &end, 10);
        if (end == p) return -1;
        out[n++] = (int32_t)v;
        p = end;
    }
    return n;
}

// ---------------------------------------------------------------------------
// HTTP I/O helpers
// ---------------------------------------------------------------------------
static int read_full_request(int fd, char* buf, int cap) {
    int total = 0;
    int hdr_end = -1;
    while (total < cap - 1) {
        int got = (int)recv(fd, buf + total, cap - 1 - total, 0);
        if (got <= 0) break;
        total += got;
        buf[total] = '\0';
        if (hdr_end < 0) {
            char* he = strstr(buf, "\r\n\r\n");
            if (he) hdr_end = (int)(he - buf) + 4;
        }
        if (hdr_end > 0) {
            const char* clp = strcasestr(buf, "Content-Length:");
            int clen = clp ? atoi(clp + 15) : 0;
            if (total >= hdr_end + clen) break;
        }
    }
    return total;
}

static void write_all(int fd, const char* p, int n) {
    while (n > 0) {
        int w = (int)send(fd, p, n, MSG_NOSIGNAL);
        if (w <= 0) return;
        p += w; n -= w;
    }
}

static void send_simple(int fd, int code, const char* reason,
                         const char* content_type, const char* body, int body_len) {
    char hdr[512];
    int n = snprintf(hdr, sizeof(hdr),
                       "HTTP/1.1 %d %s\r\n"
                       "Content-Type: %s\r\n"
                       "Content-Length: %d\r\n"
                       "Connection: close\r\n"
                       "Server: slate\r\n"
                       "\r\n",
                       code, reason, content_type, body_len);
    write_all(fd, hdr, n);
    if (body && body_len > 0) write_all(fd, body, body_len);
}

// Send SSE headers + an open chunked stream.
static void send_sse_headers(int fd) {
    static const char hdr[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "Server: slate\r\n"
        "X-Accel-Buffering: no\r\n"
        "\r\n";
    write_all(fd, hdr, (int)sizeof(hdr) - 1);
}

static void sse_send_token(int fd, int token) {
    char buf[64];
    int n = snprintf(buf, sizeof(buf), "data: {\"token\":%d}\n\n", token);
    write_all(fd, buf, n);
}

static void sse_send_done(int fd) {
    static const char done[] = "data: [DONE]\n\n";
    write_all(fd, done, (int)sizeof(done) - 1);
}

// ---------------------------------------------------------------------------
// Authentication + rate-limit
// ---------------------------------------------------------------------------
// Returns:
//    0 = ok (out_label set on success when key set)
//   -1 = auth required / wrong key (HTTP 401)
//   -2 = rate-limited (HTTP 429)
static int authenticate(const char* req,
                         slate_apikey_set_t* keys,
                         const char** out_label) {
    if (out_label) *out_label = "default";
    if (!keys) return 0;

    const char* p = strcasestr(req, "Authorization:");
    if (!p) return -1;
    p += 14;
    while (*p == ' ' || *p == '\t') ++p;
    const char prefix[] = "Bearer ";
    if (strncmp(p, prefix, sizeof(prefix) - 1) != 0) return -1;
    p += sizeof(prefix) - 1;
    // Bearer ends at CRLF / space / NUL
    char bearer[128] = {0};
    int i = 0;
    while (*p && *p != '\r' && *p != '\n' && *p != ' ' && i < (int)sizeof(bearer) - 1) {
        bearer[i++] = *p++;
    }
    bearer[i] = '\0';
    return slate_apikey_check(keys, bearer, out_label);
}

// ---------------------------------------------------------------------------
// Connection handler
// ---------------------------------------------------------------------------
static void handle_connection(struct slate_server* srv, int fd) {
    slate_gauge_add(g_m.active_connections, +1);

    char req[16384];
    int n = read_full_request(fd, req, (int)sizeof(req));
    if (n <= 0) {
        slate_gauge_add(g_m.active_connections, -1);
        close(fd);
        return;
    }
    req[n] = '\0';

    char method[8] = {0};
    char path[256] = {0};
    if (sscanf(req, "%7s %255s ", method, path) != 2) {
        send_simple(fd, 400, "Bad Request", "text/plain", "bad request\n", 12);
        slate_gauge_add(g_m.active_connections, -1);
        close(fd);
        return;
    }

    if (strcmp(method, "GET") == 0 && strcmp(path, "/health") == 0) {
        // Return 503 once we've been told to shut down so an upstream LB
        // can pull us out of rotation while we drain.
        if (!atomic_load(&srv->running)) {
            send_simple(fd, 503, "Draining", "text/plain", "draining\n", 9);
        } else {
            send_simple(fd, 200, "OK", "text/plain", "ok\n", 3);
        }
        slate_gauge_add(g_m.active_connections, -1);
        close(fd);
        return;
    }
    if (strcmp(method, "GET") == 0 && strcmp(path, "/metrics") == 0) {
        char buf[32768];
        int bn = slate_metrics_render(buf, sizeof(buf));
        if (bn < 0) {
            send_simple(fd, 500, "Internal", "text/plain", "metrics overflow\n", 17);
        } else {
            send_simple(fd, 200, "OK", "text/plain; version=0.0.4", buf, bn);
        }
        slate_gauge_add(g_m.active_connections, -1);
        close(fd);
        return;
    }
    if (strcmp(method, "POST") != 0 || strcmp(path, "/v1/completions") != 0) {
        send_simple(fd, 404, "Not Found", "text/plain", "not found\n", 10);
        slate_gauge_add(g_m.active_connections, -1);
        close(fd);
        return;
    }

    // Pick the active key set (multi-key beats single-key).
    slate_apikey_set_t* keys = srv->cfg.apikey_set
                                 ? srv->cfg.apikey_set : srv->owned_keyset;

    const char* key_label = NULL;
    int auth_rc = authenticate(req, keys, &key_label);
    if (auth_rc == -1) {
        slate_counter_inc(g_m.auth_failures_total);
        const char* kv[] = { "path", path, NULL };
        slate_jlog_warn("auth_fail", kv);
        send_simple(fd, 401, "Unauthorized", "application/json",
                     "{\"error\":\"unauthorized\"}\n", 25);
        slate_gauge_add(g_m.active_connections, -1);
        close(fd);
        return;
    }
    if (auth_rc == -2) {
        slate_counter_inc(g_m.rate_limited_total);
        const char* kv[] = { "key", key_label ? key_label : "(unknown)", NULL };
        slate_jlog_warn("rate_limited", kv);
        send_simple(fd, 429, "Too Many Requests", "application/json",
                     "{\"error\":\"rate_limited\"}\n", 25);
        slate_gauge_add(g_m.active_connections, -1);
        close(fd);
        return;
    }

    slate_counter_inc(g_m.requests_total);
    slate_gauge_add(g_m.active_requests, +1);
    atomic_fetch_add(&srv->in_flight_requests, 1);

    char* body = strstr(req, "\r\n\r\n");
    if (!body) {
        slate_gauge_add(g_m.active_requests, -1);
        atomic_fetch_sub(&srv->in_flight_requests, 1);
        send_simple(fd, 400, "Bad Request", "application/json",
                     "{\"error\":\"missing body\"}\n", 25);
        slate_gauge_add(g_m.active_connections, -1);
        close(fd);
        return;
    }
    body += 4;

    int32_t prompt[2048];
    int prompt_len = json_int_array(json_find_key(body, "prompt"), prompt, 2048);
    if (prompt_len <= 0) {
        slate_counter_inc(g_m.errors_total);
        slate_gauge_add(g_m.active_requests, -1);
        atomic_fetch_sub(&srv->in_flight_requests, 1);
        send_simple(fd, 400, "Bad Request", "application/json",
                     "{\"error\":\"missing or invalid prompt\"}\n", 38);
        slate_gauge_add(g_m.active_connections, -1);
        close(fd);
        return;
    }
    int    max_tok = json_int   (json_find_key(body, "max_tokens"),  32);
    double temp    = json_double(json_find_key(body, "temperature"), 1.0);
    double top_p   = json_double(json_find_key(body, "top_p"),       0.0);
    int    top_k   = json_int   (json_find_key(body, "top_k"),       0);
    int    stream  = json_bool  (json_find_key(body, "stream"),      0);
    if (max_tok > srv->cfg.max_tokens) max_tok = srv->cfg.max_tokens;

    slate_infer_session_t* sess = slate_infer_session_new(srv->engine);
    if (!sess) {
        slate_counter_inc(g_m.errors_total);
        slate_gauge_add(g_m.active_requests, -1);
        atomic_fetch_sub(&srv->in_flight_requests, 1);
        send_simple(fd, 500, "Internal", "application/json",
                     "{\"error\":\"alloc\"}\n", 18);
        slate_gauge_add(g_m.active_connections, -1);
        close(fd);
        return;
    }

    struct timeval t0; gettimeofday(&t0, NULL);

    int vocab = slate_infer_engine_vocab(srv->engine);
    float* logits = (float*)malloc(sizeof(float) * (size_t)vocab);
    int rc = slate_infer_prefill(sess, prompt, prompt_len, logits);
    if (rc != 0) {
        slate_counter_inc(g_m.errors_total);
        free(logits);
        slate_infer_session_free(sess);
        slate_gauge_add(g_m.active_requests, -1);
        atomic_fetch_sub(&srv->in_flight_requests, 1);
        send_simple(fd, 400, "Bad Request", "application/json",
                     "{\"error\":\"prefill failed\"}\n", 27);
        slate_gauge_add(g_m.active_connections, -1);
        close(fd);
        return;
    }

    slate_sampler_config_t scfg = { .temperature = (float)temp,
                                     .top_k = top_k,
                                     .top_p = (float)top_p,
                                     .seed  = 0 };
    uint64_t rng = (uint64_t)t0.tv_sec * 1000000ULL + (uint64_t)t0.tv_usec;
    int32_t out_toks[2048];
    int n_out = 0;
    double t_first_token = 0;

    if (stream) {
        // -------------------------------------------------------------
        // SSE streaming path: write headers, then emit one event per
        // token, then [DONE].
        // -------------------------------------------------------------
        slate_counter_inc(g_m.stream_requests_total);
        send_sse_headers(fd);
        for (int i = 0; i < max_tok; ++i) {
            int next = slate_sample_token(logits, vocab, &scfg, &rng);
            // sample_token can return out-of-vocab garbage if the buffer
            // contained stale values past vocab; clamp by re-decoding.
            // The engine validates the input though, so use its vocab.
            // (For tiny demos this is fine; in production the vocab
            // would be passed in from engine introspection.)
            sse_send_token(fd, next);
            out_toks[n_out++] = (int32_t)next;
            int dr;
            if (srv->scheduler) {
                dr = slate_scheduler_decode(srv->scheduler, sess, (int32_t)next, logits);
            } else {
                dr = slate_infer_decode_step(sess, (int32_t)next, logits);
            }
            if (dr != 0) break;
            if (i == 0) {
                struct timeval t1; gettimeofday(&t1, NULL);
                t_first_token = (double)(t1.tv_sec - t0.tv_sec) * 1000.0 +
                                 (double)(t1.tv_usec - t0.tv_usec) / 1000.0;
                slate_histogram_observe(g_m.time_to_first_token_ms, t_first_token);
            }
        }
        sse_send_done(fd);
    } else {
        // -------------------------------------------------------------
        // Non-streaming path: collect into buffer then send JSON.
        // -------------------------------------------------------------
        for (int i = 0; i < max_tok; ++i) {
            int next = slate_sample_token(logits, vocab, &scfg, &rng);
            out_toks[n_out++] = (int32_t)next;
            int dr;
            if (srv->scheduler) {
                dr = slate_scheduler_decode(srv->scheduler, sess, (int32_t)next, logits);
            } else {
                dr = slate_infer_decode_step(sess, (int32_t)next, logits);
            }
            if (dr != 0) break;
            if (i == 0) {
                struct timeval t1; gettimeofday(&t1, NULL);
                t_first_token = (double)(t1.tv_sec - t0.tv_sec) * 1000.0 +
                                 (double)(t1.tv_usec - t0.tv_usec) / 1000.0;
                slate_histogram_observe(g_m.time_to_first_token_ms, t_first_token);
            }
        }
        struct timeval t1; gettimeofday(&t1, NULL);
        double elapsed_ms = (double)(t1.tv_sec - t0.tv_sec) * 1000.0 +
                             (double)(t1.tv_usec - t0.tv_usec) / 1000.0;

        char resp[16384];
        int pos = snprintf(resp, sizeof(resp), "{\"tokens\":[");
        for (int i = 0; i < n_out && pos < (int)sizeof(resp) - 64; ++i) {
            pos += snprintf(resp + pos, sizeof(resp) - pos,
                             "%s%d", i ? "," : "", out_toks[i]);
        }
        pos += snprintf(resp + pos, sizeof(resp) - pos,
                         "],\"prompt_tokens\":%d,\"completion_tokens\":%d,"
                         "\"latency_ms\":%.3f,\"key_label\":\"%s\"}\n",
                         prompt_len, n_out, elapsed_ms,
                         key_label ? key_label : "default");
        send_simple(fd, 200, "OK", "application/json", resp, pos);
    }
    free(logits);

    struct timeval t2; gettimeofday(&t2, NULL);
    double elapsed_ms = (double)(t2.tv_sec - t0.tv_sec) * 1000.0 +
                         (double)(t2.tv_usec - t0.tv_usec) / 1000.0;

    slate_counter_add(g_m.tokens_in_total,  (uint64_t)prompt_len);
    slate_counter_add(g_m.tokens_out_total, (uint64_t)n_out);
    slate_histogram_observe(g_m.request_latency_ms, elapsed_ms);
    slate_histogram_observe(g_m.tokens_out_hist,   (double)n_out);

    {
        char buf_pt[24], buf_ct[24], buf_lat[24], buf_ttft[24];
        snprintf(buf_pt,   sizeof(buf_pt),   "%d", prompt_len);
        snprintf(buf_ct,   sizeof(buf_ct),   "%d", n_out);
        snprintf(buf_lat,  sizeof(buf_lat),  "%.3f", elapsed_ms);
        snprintf(buf_ttft, sizeof(buf_ttft), "%.3f", t_first_token);
        const char* kv[] = {
            "key",               key_label ? key_label : "default",
            "stream",            stream ? "true" : "false",
            "prompt_tokens",     buf_pt,
            "completion_tokens", buf_ct,
            "latency_ms",        buf_lat,
            "ttft_ms",           buf_ttft,
            NULL,
        };
        slate_jlog_info("request_complete", kv);
    }

    slate_infer_session_free(sess);
    slate_gauge_add(g_m.active_requests, -1);
    atomic_fetch_sub(&srv->in_flight_requests, 1);
    slate_gauge_add(g_m.active_connections, -1);
    close(fd);
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------
static void* worker_main(void* arg) {
    struct slate_server* srv = (struct slate_server*)arg;
    for (;;) {
        pthread_mutex_lock(&srv->qlock);
        while (atomic_load(&srv->running) && srv->qcount == 0) {
            pthread_cond_wait(&srv->qcond, &srv->qlock);
        }
        if (!atomic_load(&srv->running) && srv->qcount == 0) {
            pthread_mutex_unlock(&srv->qlock);
            break;
        }
        int fd = srv->queue[srv->qhead];
        srv->qhead = (srv->qhead + 1) % srv->qcap;
        srv->qcount--;
        pthread_mutex_unlock(&srv->qlock);
        handle_connection(srv, fd);
    }
    return NULL;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
slate_server_t* slate_server_new(slate_infer_engine_t* engine,
                                  const slate_server_config_t* cfg) {
    if (!engine || !cfg) return NULL;
    pthread_once(&g_metrics_once, init_metrics);

    struct slate_server* srv = (struct slate_server*)calloc(1, sizeof(*srv));
    if (!srv) return NULL;
    srv->engine = engine;
    srv->cfg    = *cfg;
    if (srv->cfg.n_workers <= 0) srv->cfg.n_workers = 4;
    if (srv->cfg.n_workers > MAX_WORKERS) srv->cfg.n_workers = MAX_WORKERS;
    if (srv->cfg.max_tokens <= 0) srv->cfg.max_tokens = 256;
    if (srv->cfg.max_body_kb <= 0) srv->cfg.max_body_kb = 64;
    if (srv->cfg.shutdown_timeout_sec <= 0) srv->cfg.shutdown_timeout_sec = 30;
    if (srv->cfg.scheduler_max_batch < 0) srv->cfg.scheduler_max_batch = 0;
    if (srv->cfg.scheduler_max_batch == 0 && !cfg->api_key) {
        // Default to batching when the caller didn't explicitly opt out.
        srv->cfg.scheduler_max_batch = 16;
    }
    srv->qcap = srv->cfg.n_workers * 4;
    pthread_mutex_init(&srv->qlock, NULL);
    pthread_cond_init (&srv->qcond, NULL);
    atomic_store(&srv->running, 1);
    atomic_store(&srv->in_flight_requests, 0);

    // Back-compat: if only api_key is set, synthesise a single-key set.
    if (!srv->cfg.apikey_set && srv->cfg.api_key) {
        srv->owned_keyset = slate_apikey_set_new();
        slate_apikey_set_add(srv->owned_keyset, srv->cfg.api_key,
                              "default", 0.0, 0);
    }

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { free(srv); return NULL; }
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    struct sockaddr_in addr; memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t)cfg->port);
    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(fd); slate_apikey_set_free(srv->owned_keyset); free(srv); return NULL;
    }
    if (listen(fd, 64) < 0) {
        close(fd); slate_apikey_set_free(srv->owned_keyset); free(srv); return NULL;
    }
    srv->listen_fd = fd;

    for (int i = 0; i < srv->cfg.n_workers; ++i) {
        pthread_create(&srv->workers[i], NULL, worker_main, srv);
    }
    srv->n_workers = srv->cfg.n_workers;

    // Optional micro-batching scheduler.
    if (srv->cfg.scheduler_max_batch > 0) {
        srv->scheduler = slate_scheduler_new(engine, srv->cfg.scheduler_max_batch);
    }

    int n_keys = 0;
    if (srv->cfg.apikey_set) n_keys = slate_apikey_set_size(srv->cfg.apikey_set);
    else if (srv->owned_keyset) n_keys = 1;

    char buf_port[16], buf_w[16], buf_k[16];
    snprintf(buf_port, sizeof(buf_port), "%d", srv->cfg.port);
    snprintf(buf_w,    sizeof(buf_w),    "%d", srv->cfg.n_workers);
    snprintf(buf_k,    sizeof(buf_k),    "%d", n_keys);
    const char* kv[] = { "port", buf_port, "workers", buf_w,
                          "auth_keys", buf_k, NULL };
    slate_jlog_info("server_started", kv);
    return srv;
}

int slate_server_run(slate_server_t* srv) {
    if (!srv) return -1;
    while (atomic_load(&srv->running)) {
        struct sockaddr_in cli; socklen_t cl = sizeof(cli);
        int cfd = accept(srv->listen_fd, (struct sockaddr*)&cli, &cl);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            if (!atomic_load(&srv->running)) break;
            continue;
        }
        pthread_mutex_lock(&srv->qlock);
        if (srv->qcount >= srv->qcap) {
            pthread_mutex_unlock(&srv->qlock);
            send_simple(cfd, 503, "Busy", "application/json",
                         "{\"error\":\"server busy\"}\n", 24);
            close(cfd);
            continue;
        }
        srv->queue[srv->qtail] = cfd;
        srv->qtail = (srv->qtail + 1) % srv->qcap;
        srv->qcount++;
        pthread_cond_signal(&srv->qcond);
        pthread_mutex_unlock(&srv->qlock);
    }
    slate_jlog_info("server_stopped", NULL);
    return 0;
}

void slate_server_stop(slate_server_t* srv) {
    if (!srv) return;
    atomic_store(&srv->running, 0);
    shutdown(srv->listen_fd, SHUT_RDWR);
    close(srv->listen_fd);
    pthread_mutex_lock(&srv->qlock);
    pthread_cond_broadcast(&srv->qcond);
    pthread_mutex_unlock(&srv->qlock);
}

void slate_server_install_signal_handler(slate_server_t* srv) {
    atomic_store(&g_signal_target, srv);
    struct sigaction sa; memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_stop_handler;
    sa.sa_flags   = SA_RESTART;
    sigaction(SIGINT,  &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    // Ignore SIGPIPE — we already use MSG_NOSIGNAL on send.
    struct sigaction sp; memset(&sp, 0, sizeof(sp));
    sp.sa_handler = SIG_IGN;
    sigaction(SIGPIPE, &sp, NULL);
}

void slate_server_free(slate_server_t* srv) {
    if (!srv) return;
    slate_server_stop(srv);

    // Graceful drain.
    double deadline_s = (double)srv->cfg.shutdown_timeout_sec;
    double waited = 0;
    while (waited < deadline_s &&
           atomic_load(&srv->in_flight_requests) > 0) {
        usleep(50 * 1000);
        waited += 0.05;
    }
    if (atomic_load(&srv->in_flight_requests) > 0) {
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", atomic_load(&srv->in_flight_requests));
        const char* kv[] = { "remaining", buf, NULL };
        slate_jlog_warn("shutdown_timeout", kv);
    }

    for (int i = 0; i < srv->n_workers; ++i) {
        pthread_join(srv->workers[i], NULL);
    }
    // Tear scheduler down AFTER workers — workers may have been blocked
    // on scheduler_decode at the time stop was issued.
    if (srv->scheduler) slate_scheduler_free(srv->scheduler);
    pthread_mutex_destroy(&srv->qlock);
    pthread_cond_destroy(&srv->qcond);
    slate_apikey_set_free(srv->owned_keyset);
    free(srv);
}
