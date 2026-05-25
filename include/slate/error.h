// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// error.h — error handling utilities.
//
// Slate's error model: functions that can fail return slate_status_t. Callers
// check the return code. Out-parameters are filled only when the return is
// SLATE_OK.

#ifndef SLATE_ERROR_H
#define SLATE_ERROR_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns a thread-local detail message describing the most recent error on
// this thread. Cleared on the next SLATE_OK-returning call. Useful when the
// status code alone is not specific enough.
const char* slate_last_error(void);

// Internal use by library code; not part of stable API. Formats a message and
// stores it in the thread-local error slot, then returns `status`.
slate_status_t slate_set_error(slate_status_t status, const char* fmt, ...);

// Convenience macros for hot paths.
#define SLATE_CHECK(expr) \
    do { slate_status_t _s = (expr); if (_s != SLATE_OK) return _s; } while (0)

#define SLATE_RETURN_IF_NULL(p) \
    do { if (!(p)) return slate_set_error(SLATE_ERR_INVALID_ARGUMENT, \
                                          "%s is NULL", #p); } while (0)

#ifdef __cplusplus
}
#endif

#endif // SLATE_ERROR_H
