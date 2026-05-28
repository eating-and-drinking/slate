// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// jlog.c — structured JSON logger.  Single-mutex serialisation
// around fputs() so log lines stay non-interleaved between threads.

#include "slate/jlog.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static slate_jlog_level_t  g_min_level = SLATE_JLOG_INFO;
static FILE*               g_fp        = NULL;   // lazy stderr
static pthread_mutex_t     g_lock      = PTHREAD_MUTEX_INITIALIZER;

static const char* level_str(slate_jlog_level_t l) {
    switch (l) {
        case SLATE_JLOG_DEBUG: return "debug";
        case SLATE_JLOG_INFO:  return "info";
        case SLATE_JLOG_WARN:  return "warn";
        case SLATE_JLOG_ERROR: return "error";
    }
    return "info";
}

void slate_jlog_set_level(slate_jlog_level_t level) {
    pthread_mutex_lock(&g_lock);
    g_min_level = level;
    pthread_mutex_unlock(&g_lock);
}

void slate_jlog_set_output(void* fp) {
    pthread_mutex_lock(&g_lock);
    g_fp = (FILE*)fp;
    pthread_mutex_unlock(&g_lock);
}

// Append a JSON-escaped string to buf at position *pos with capacity cap.
static int json_escape(char* buf, size_t cap, int pos, const char* s) {
    if (!s) s = "";
    for (; *s; ++s) {
        char c = *s;
        if ((size_t)pos + 6 >= cap) return -1;
        switch (c) {
            case '"':  buf[pos++] = '\\'; buf[pos++] = '"'; break;
            case '\\': buf[pos++] = '\\'; buf[pos++] = '\\'; break;
            case '\n': buf[pos++] = '\\'; buf[pos++] = 'n'; break;
            case '\r': buf[pos++] = '\\'; buf[pos++] = 'r'; break;
            case '\t': buf[pos++] = '\\'; buf[pos++] = 't'; break;
            default:
                if ((unsigned char)c < 0x20) {
                    pos += snprintf(buf + pos, cap - pos, "\\u%04x", (unsigned)c);
                } else {
                    buf[pos++] = c;
                }
        }
    }
    return pos;
}

static int append_lit(char* buf, size_t cap, int pos, const char* lit) {
    size_t l = strlen(lit);
    if ((size_t)pos + l >= cap) return -1;
    memcpy(buf + pos, lit, l);
    return pos + (int)l;
}

void slate_jlog(slate_jlog_level_t level,
                const char* event,
                const char* const* kv) {
    pthread_mutex_lock(&g_lock);
    if (level < g_min_level) { pthread_mutex_unlock(&g_lock); return; }
    FILE* fp = g_fp ? g_fp : stderr;
    pthread_mutex_unlock(&g_lock);

    char line[2048];
    int pos = 0;
    // {"ts":"...","level":"...","event":"...","fields":{
    time_t now = time(NULL);
    struct tm tmv;
#if defined(_WIN32)
    gmtime_s(&tmv, &now);
#else
    gmtime_r(&now, &tmv);
#endif
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", &tmv);
    pos = append_lit(line, sizeof(line), pos, "{\"ts\":\"");
    if (pos < 0) goto done;
    pos = append_lit(line, sizeof(line), pos, ts);
    if (pos < 0) goto done;
    pos = append_lit(line, sizeof(line), pos, "\",\"level\":\"");
    if (pos < 0) goto done;
    pos = append_lit(line, sizeof(line), pos, level_str(level));
    if (pos < 0) goto done;
    pos = append_lit(line, sizeof(line), pos, "\",\"event\":\"");
    if (pos < 0) goto done;
    pos = json_escape(line, sizeof(line), pos, event ? event : "");
    if (pos < 0) goto done;
    pos = append_lit(line, sizeof(line), pos, "\"");
    if (pos < 0) goto done;
    if (kv && kv[0]) {
        pos = append_lit(line, sizeof(line), pos, ",\"fields\":{");
        if (pos < 0) goto done;
        int first = 1;
        for (int i = 0; kv[i] != NULL && kv[i+1] != NULL; i += 2) {
            if (!first) {
                pos = append_lit(line, sizeof(line), pos, ",");
                if (pos < 0) goto done;
            }
            first = 0;
            pos = append_lit(line, sizeof(line), pos, "\"");
            if (pos < 0) goto done;
            pos = json_escape(line, sizeof(line), pos, kv[i]);
            if (pos < 0) goto done;
            pos = append_lit(line, sizeof(line), pos, "\":\"");
            if (pos < 0) goto done;
            pos = json_escape(line, sizeof(line), pos, kv[i+1]);
            if (pos < 0) goto done;
            pos = append_lit(line, sizeof(line), pos, "\"");
            if (pos < 0) goto done;
        }
        pos = append_lit(line, sizeof(line), pos, "}");
        if (pos < 0) goto done;
    }
    pos = append_lit(line, sizeof(line), pos, "}\n");
done:
    if (pos < 0) return;   // overflow: silently drop
    line[pos] = '\0';
    pthread_mutex_lock(&g_lock);
    if (fp) {
        fwrite(line, 1, (size_t)pos, fp);
        fflush(fp);
    }
    pthread_mutex_unlock(&g_lock);
}

void slate_jlog_info (const char* event, const char* const* kv) {
    slate_jlog(SLATE_JLOG_INFO,  event, kv);
}
void slate_jlog_warn (const char* event, const char* const* kv) {
    slate_jlog(SLATE_JLOG_WARN,  event, kv);
}
void slate_jlog_error(const char* event, const char* const* kv) {
    slate_jlog(SLATE_JLOG_ERROR, event, kv);
}
