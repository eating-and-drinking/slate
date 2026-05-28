// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// jlog.h — structured (JSON) logging for the inference server.
//
// One JSON object per line on stderr.  Format:
//   {"ts":"2026-05-27T12:34:56Z","level":"info","event":"request",
//    "fields":{"client":"1.2.3.4","tokens":12,"latency_ms":45.2}}
//
// Thread-safe via a single mutex around the line emission; the
// payload assembly happens in a stack buffer per call.
//
// Designed to be ingested by Loki / OpenSearch / Splunk without
// further parsing.

#ifndef SLATE_JLOG_H
#define SLATE_JLOG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SLATE_JLOG_DEBUG = 0,
    SLATE_JLOG_INFO  = 1,
    SLATE_JLOG_WARN  = 2,
    SLATE_JLOG_ERROR = 3
} slate_jlog_level_t;

// Set the minimum level to emit (default: INFO).
void slate_jlog_set_level(slate_jlog_level_t level);

// Set the output FILE* (default: stderr).  Pass NULL to disable
// emission entirely (useful in tests).
void slate_jlog_set_output(void* fp);

// Emit one log line.  `event` is a stable short identifier (e.g.
// "request_complete", "auth_fail"); `kv_pairs` is a NULL-terminated
// list of {"key", "value"} string pairs that will be put into the
// JSON "fields" object.  Pass NULL kv_pairs for events with no
// extra structured data.
//
// Example:
//   const char* kv[] = { "client", "1.2.3.4",
//                        "tokens", "47",
//                        "latency_ms", "12.3",
//                        NULL };
//   slate_jlog(SLATE_JLOG_INFO, "request_complete", kv);
void slate_jlog(slate_jlog_level_t level,
                const char* event,
                const char* const* kv_pairs);

// Convenience wrappers.  These are not formatted printf calls
// — keep the message field of `event` short.
void slate_jlog_info  (const char* event, const char* const* kv);
void slate_jlog_warn  (const char* event, const char* const* kv);
void slate_jlog_error (const char* event, const char* const* kv);

#ifdef __cplusplus
}
#endif

#endif // SLATE_JLOG_H
