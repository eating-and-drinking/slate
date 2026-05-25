// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// log.c — global logging.

#include "slate/runtime.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static slate_log_level_t g_level = SLATE_LOG_INFO;

void slate_log_set_level(slate_log_level_t level) { g_level = level; }
slate_log_level_t slate_log_get_level(void) { return g_level; }

static const char* level_name(slate_log_level_t lv) {
    switch (lv) {
        case SLATE_LOG_TRACE: return "TRACE";
        case SLATE_LOG_DEBUG: return "DEBUG";
        case SLATE_LOG_INFO:  return "INFO";
        case SLATE_LOG_WARN:  return "WARN";
        case SLATE_LOG_ERROR: return "ERROR";
        default:              return "?";
    }
}

void slate_log_message(slate_log_level_t level, const char* file, int line,
                       const char* fmt, ...) {
    if (level < g_level) return;

    const char* base = strrchr(file, '/');
#ifdef _WIN32
    const char* baseb = strrchr(file, '\\');
    if (baseb && baseb > base) base = baseb;
#endif
    base = base ? base + 1 : file;

    fprintf(stderr, "[%-5s %s:%d] ", level_name(level), base, line);

    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);

    fputc('\n', stderr);
}
