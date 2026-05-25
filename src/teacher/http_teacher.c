#define _POSIX_C_SOURCE 200809L
// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#include "slate/http_teacher.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum { PROV_OPENAI, PROV_ANTHROPIC } provider_t;

struct slate_http_teacher {
    provider_t provider;
    char* model;
    char* api_key;
    slate_http_transport_fn transport;
    void* transport_ud;
};

void slate_http_response_free(slate_http_response_t* r) {
    if (!r) return;
    free(r->text); free(r->token_ids); free(r->top_logits); free(r->error);
    memset(r, 0, sizeof(*r));
}

static slate_http_teacher_t* new_with(provider_t p, const char* model, const char* key) {
    slate_http_teacher_t* t = (slate_http_teacher_t*)calloc(1, sizeof(*t));
    t->provider = p;
    t->model = strdup(model ? model : "");
    t->api_key = strdup(key ? key : "");
    return t;
}
slate_http_teacher_t* slate_http_teacher_openai_new(const char* m, const char* k)  { return new_with(PROV_OPENAI, m, k); }
slate_http_teacher_t* slate_http_teacher_anthropic_new(const char* m, const char* k){ return new_with(PROV_ANTHROPIC, m, k); }

void slate_http_teacher_set_transport(slate_http_teacher_t* t,
                                       slate_http_transport_fn fn, void* ud) {
    if (!t) return; t->transport = fn; t->transport_ud = ud;
}

slate_http_response_t slate_http_teacher_generate(slate_http_teacher_t* t,
                                                   const char* prompt, int max_tokens) {
    slate_http_response_t r; memset(&r, 0, sizeof(r));
    if (!t || !prompt) { r.error = strdup("invalid arguments"); return r; }
    if (!t->transport) {
        r.error = strdup("HTTP transport not configured (call slate_http_teacher_set_transport)");
        return r;
    }
    // Build a small JSON request body. We avoid a full JSON library here;
    // the transport callback may build a richer one if needed.
    const char* endpoint = (t->provider == PROV_OPENAI)
        ? "https://api.openai.com/v1/chat/completions"
        : "https://api.anthropic.com/v1/messages";
    char body[2048];
    snprintf(body, sizeof(body),
        "{\"model\":\"%s\",\"max_tokens\":%d,\"prompt\":\"%.1500s\"}",
        t->model, max_tokens, prompt);

    char* resp = NULL;
    int rc = t->transport(endpoint, body, &resp, t->transport_ud);
    if (rc != 0 || !resp) {
        r.error = strdup("transport failed");
        free(resp);
        return r;
    }
    // For the interface stub, we just return the raw response as text.
    // A real client would JSON-parse to extract content + logprobs.
    r.text = resp;
    return r;
}

void slate_http_teacher_destroy(slate_http_teacher_t* t) {
    if (!t) return; free(t->model); free(t->api_key); free(t);
}
