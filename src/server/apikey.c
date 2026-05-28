// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// apikey.c — multi-key auth + per-key token-bucket rate limiting.
// One mutex per key entry so that bursts on key A don't serialise
// against requests carrying key B.  JSON parser here is intentionally
// dumb — it's just enough to read the documented schema, not a
// general-purpose decoder.

#include "slate/apikey.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

typedef struct apikey {
    char*   secret;
    char*   label;       // owns, or NULL → derived from secret prefix
    double  rps;         // 0 = unlimited
    int     burst;       // bucket cap
    double  tokens;      // current bucket level (double for sub-tick refill)
    double  last_us;     // last refill time (microseconds since epoch)
    pthread_mutex_t lock;
} apikey_t;

struct slate_apikey_set {
    apikey_t* entries;
    int       n_entries;
    int       cap_entries;
    pthread_mutex_t lock;     // protects the entries array itself
};

static double now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}

slate_apikey_set_t* slate_apikey_set_new(void) {
    slate_apikey_set_t* s = (slate_apikey_set_t*)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->cap_entries = 16;
    s->entries = (apikey_t*)calloc((size_t)s->cap_entries, sizeof(apikey_t));
    if (!s->entries) { free(s); return NULL; }
    pthread_mutex_init(&s->lock, NULL);
    return s;
}

void slate_apikey_set_free(slate_apikey_set_t* set) {
    if (!set) return;
    for (int i = 0; i < set->n_entries; ++i) {
        pthread_mutex_destroy(&set->entries[i].lock);
        free(set->entries[i].secret);
        free(set->entries[i].label);
    }
    free(set->entries);
    pthread_mutex_destroy(&set->lock);
    free(set);
}

int slate_apikey_set_add(slate_apikey_set_t* set,
                          const char* key,
                          const char* label,
                          double rps,
                          int    burst) {
    if (!set || !key || !*key) return -1;
    pthread_mutex_lock(&set->lock);
    // Capacity check
    if (set->n_entries >= set->cap_entries) {
        int nc = set->cap_entries * 2;
        apikey_t* ne = (apikey_t*)realloc(set->entries, (size_t)nc * sizeof(apikey_t));
        if (!ne) { pthread_mutex_unlock(&set->lock); return -1; }
        set->entries = ne;
        set->cap_entries = nc;
    }
    // Dup check
    for (int i = 0; i < set->n_entries; ++i) {
        if (strcmp(set->entries[i].secret, key) == 0) {
            pthread_mutex_unlock(&set->lock);
            return -1;
        }
    }
    apikey_t* e = &set->entries[set->n_entries];
    memset(e, 0, sizeof(*e));
    e->secret = strdup(key);
    e->label  = label ? strdup(label) : NULL;
    e->rps    = rps;
    e->burst  = burst;
    e->tokens = (double)burst;     // start with a full bucket
    e->last_us = now_us();
    pthread_mutex_init(&e->lock, NULL);
    if (!e->secret || (label && !e->label)) {
        free(e->secret); free(e->label);
        pthread_mutex_destroy(&e->lock);
        pthread_mutex_unlock(&set->lock);
        return -1;
    }
    set->n_entries++;
    pthread_mutex_unlock(&set->lock);
    return 0;
}

int slate_apikey_set_size(const slate_apikey_set_t* set) {
    if (!set) return 0;
    pthread_mutex_lock((pthread_mutex_t*)&set->lock);
    int n = set->n_entries;
    pthread_mutex_unlock((pthread_mutex_t*)&set->lock);
    return n;
}

// ---------------------------------------------------------------------------
// Tiny JSON parser for the config file schema.
// ---------------------------------------------------------------------------
// Accepts:  [ { "key":"...", "label":"...", "rps":N, "burst":M }, ... ]
// Skips comments (a leading '#' line is dropped before parsing).
// On any parse failure returns the number of keys loaded so far so
// the operator can spot the bad entry by line context in the log.

static const char* skip_ws(const char* p) {
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    return p;
}

static const char* parse_string(const char* p, char** out) {
    p = skip_ws(p);
    if (*p != '"') return NULL;
    ++p;
    const char* start = p;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) p += 2;
        else ++p;
    }
    if (*p != '"') return NULL;
    int n = (int)(p - start);
    char* s = (char*)malloc((size_t)n + 1);
    if (!s) return NULL;
    memcpy(s, start, (size_t)n);
    s[n] = '\0';
    *out = s;
    return p + 1;
}

static const char* parse_number(const char* p, double* out) {
    p = skip_ws(p);
    char* end;
    *out = strtod(p, &end);
    if (end == p) return NULL;
    return end;
}

int slate_apikey_set_load(slate_apikey_set_t* set, const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); return -1; }
    fread(buf, 1, (size_t)sz, fp);
    buf[sz] = '\0';
    fclose(fp);

    // Strip line-comments starting with '#'
    char* w = buf;
    int in_str = 0;
    for (char* r = buf; *r; ++r) {
        if (in_str) {
            if (*r == '"' && r > buf && r[-1] != '\\') in_str = 0;
            *w++ = *r;
        } else if (*r == '"') {
            in_str = 1; *w++ = *r;
        } else if (*r == '#') {
            while (*r && *r != '\n') ++r;
            if (*r == '\n') *w++ = '\n';
        } else {
            *w++ = *r;
        }
    }
    *w = '\0';

    int loaded = 0;
    const char* p = skip_ws(buf);
    if (*p != '[') { free(buf); return -1; }
    ++p;
    while (1) {
        p = skip_ws(p);
        if (*p == ']') break;
        if (*p != '{') { free(buf); return loaded; }
        ++p;
        char* key = NULL;
        char* label = NULL;
        double rps = 0;
        int    burst = 0;
        while (1) {
            p = skip_ws(p);
            if (*p == '}') break;
            char* field = NULL;
            const char* np = parse_string(p, &field);
            if (!np) { free(field); free(key); free(label); free(buf); return loaded; }
            p = np;
            p = skip_ws(p);
            if (*p != ':') { free(field); free(key); free(label); free(buf); return loaded; }
            ++p;
            if (strcmp(field, "key") == 0) {
                p = parse_string(p, &key);
            } else if (strcmp(field, "label") == 0) {
                p = parse_string(p, &label);
            } else if (strcmp(field, "rps") == 0) {
                double v; p = parse_number(p, &v); rps = v;
            } else if (strcmp(field, "burst") == 0) {
                double v; p = parse_number(p, &v); burst = (int)v;
            } else {
                // Unknown field — skip the value (string or number)
                p = skip_ws(p);
                if (*p == '"') {
                    char* dummy = NULL; p = parse_string(p, &dummy); free(dummy);
                } else {
                    double v; p = parse_number(p, &v);
                }
            }
            free(field);
            if (!p) { free(key); free(label); free(buf); return loaded; }
            p = skip_ws(p);
            if (*p == ',') { ++p; continue; }
            if (*p == '}') break;
        }
        ++p;     // past '}'
        if (key) {
            if (slate_apikey_set_add(set, key, label, rps, burst > 0 ? burst : 1) == 0) {
                loaded++;
            }
        }
        free(key); free(label);
        p = skip_ws(p);
        if (*p == ',') { ++p; continue; }
        if (*p == ']') break;
    }
    free(buf);
    return loaded;
}

int slate_apikey_check(slate_apikey_set_t* set,
                        const char* bearer,
                        const char** out_label) {
    if (!set || !bearer) return -1;
    apikey_t* e = NULL;
    pthread_mutex_lock(&set->lock);
    for (int i = 0; i < set->n_entries; ++i) {
        if (strcmp(set->entries[i].secret, bearer) == 0) {
            e = &set->entries[i]; break;
        }
    }
    pthread_mutex_unlock(&set->lock);
    if (!e) return -1;

    pthread_mutex_lock(&e->lock);
    if (e->rps > 0 && e->burst > 0) {
        // Refill the bucket.
        double now = now_us();
        double dt_s = (now - e->last_us) * 1e-6;
        e->last_us = now;
        e->tokens += dt_s * e->rps;
        if (e->tokens > (double)e->burst) e->tokens = (double)e->burst;
        if (e->tokens < 1.0) {
            pthread_mutex_unlock(&e->lock);
            return -2;
        }
        e->tokens -= 1.0;
    }
    if (out_label) {
        *out_label = e->label ? e->label : e->secret;
    }
    pthread_mutex_unlock(&e->lock);
    return 0;
}
