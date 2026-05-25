// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// executor.h — sandboxed code execution.
//
// This interface is for running code that the language model generated. That
// code is *untrusted input* and must be isolated from the host filesystem,
// network, and process space.
//
// Four implementations are planned:
//   - WasmtimePyodideExecutor: cross-platform, default
//   - SubprocessSandboxExecutor: Linux/macOS, faster, uses seccomp + ulimits
//   - DockerExecutor: dev/CI only, heavy but well-known
//   - AndroidIsolatedExecutor: Android-specific
//
// SECURITY NOTE: see SECURITY.md for the threat model.
//
// IMPLEMENTATION STATUS: M7.

#ifndef SLATE_EXECUTOR_H
#define SLATE_EXECUTOR_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_executor slate_executor_t;

typedef struct slate_exec_limits {
    int    timeout_ms;       // wall clock, default 5000
    int    memory_mb;        // hard limit, default 256
    bool   allow_network;    // ALWAYS FALSE in default policy
    bool   allow_filesystem; // if true, /tmp/sandbox is mounted writable
    int    cpu_limit_pct;    // 0 = no limit, else 1-100
    const char* language;    // "python", "c", "rust", "javascript"
} slate_exec_limits_t;

typedef struct slate_exec_result {
    bool    compiled;
    bool    ran;
    bool    timed_out;
    bool    oom;
    int     exit_code;
    char*   stdout_data;
    size_t  stdout_size;
    char*   stderr_data;
    size_t  stderr_size;
    double  wall_time_ms;
    double  memory_peak_mb;
} slate_exec_result_t;

void slate_exec_result_free(slate_exec_result_t* r);

struct slate_executor {
    const char* backend_name;

    slate_status_t (*execute)(slate_executor_t* self,
                              const char* code,
                              const char* stdin_input,
                              const slate_exec_limits_t* limits,
                              slate_exec_result_t* out);

    void (*destroy)(slate_executor_t* self);
    void* user_data;
};

slate_executor_t* slate_executor_wasmtime_pyodide_new(void);
slate_executor_t* slate_executor_subprocess_sandbox_new(void);
slate_executor_t* slate_executor_docker_new(const char* image);

#ifdef __cplusplus
}
#endif

#endif // SLATE_EXECUTOR_H
