// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// error.c — error string table and thread-local detail message.

#include "slate/error.h"
#include "slate/types.h"
#include "slate/version.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

const char* slate_version_string(void) { return "slate " SLATE_VERSION_STRING; }
int slate_version_num(void) { return SLATE_VERSION_NUM; }

const char* slate_status_string(slate_status_t s) {
    switch (s) {
        case SLATE_OK:                     return "ok";
        case SLATE_ERR_INVALID_ARGUMENT:   return "invalid argument";
        case SLATE_ERR_OUT_OF_MEMORY:      return "out of memory";
        case SLATE_ERR_SHAPE_MISMATCH:     return "shape mismatch";
        case SLATE_ERR_DTYPE_MISMATCH:     return "dtype mismatch";
        case SLATE_ERR_NOT_IMPLEMENTED:    return "not implemented";
        case SLATE_ERR_IO:                 return "I/O error";
        case SLATE_ERR_INTERNAL:           return "internal error";
        case SLATE_ERR_RUNTIME_MODE:       return "wrong runtime mode";
        case SLATE_ERR_NO_GRAD:            return "no gradient";
        default:                           return "unknown";
    }
}

const char* slate_dtype_name(slate_dtype_t dt) {
    switch (dt) {
        case SLATE_DTYPE_F32:  return "f32";
        case SLATE_DTYPE_F16:  return "f16";
        case SLATE_DTYPE_BF16: return "bf16";
        case SLATE_DTYPE_I32:  return "i32";
        case SLATE_DTYPE_I8:   return "i8";
        case SLATE_DTYPE_Q8_0: return "q8_0";
        case SLATE_DTYPE_Q4_0: return "q4_0";
        case SLATE_DTYPE_Q4_K: return "q4_k";
        default:               return "?";
    }
}

size_t slate_dtype_size(slate_dtype_t dt) {
    switch (dt) {
        case SLATE_DTYPE_F32:  return 4;
        case SLATE_DTYPE_F16:  return 2;
        case SLATE_DTYPE_BF16: return 2;
        case SLATE_DTYPE_I32:  return 4;
        case SLATE_DTYPE_I8:   return 1;
        // Block-quantized: implementation arrives in M5. Values here are
        // placeholders so the table is complete; do not rely on them.
        case SLATE_DTYPE_Q8_0: return 34;   // 32 elements + scale
        case SLATE_DTYPE_Q4_0: return 18;   // 32 4-bit nibbles + scale
        case SLATE_DTYPE_Q4_K: return 144;  // super-block of 256 -> 144 bytes
        default:               return 0;
    }
}

bool slate_dtype_is_float(slate_dtype_t dt) {
    return dt == SLATE_DTYPE_F32 || dt == SLATE_DTYPE_F16 || dt == SLATE_DTYPE_BF16;
}

bool slate_dtype_is_quantized(slate_dtype_t dt) {
    return dt == SLATE_DTYPE_Q8_0 || dt == SLATE_DTYPE_Q4_0 || dt == SLATE_DTYPE_Q4_K;
}

// =============================================================================
// Thread-local error detail.
// =============================================================================
#if defined(_MSC_VER)
    #define SLATE_THREAD_LOCAL __declspec(thread)
#else
    #define SLATE_THREAD_LOCAL __thread
#endif

static SLATE_THREAD_LOCAL char g_last_error[256];

const char* slate_last_error(void) {
    return g_last_error[0] ? g_last_error : "(no error)";
}

slate_status_t slate_set_error(slate_status_t status, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_last_error, sizeof(g_last_error), fmt, ap);
    va_end(ap);
    return status;
}
