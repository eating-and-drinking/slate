// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// code_executor.h — sandboxed code execution for code-RL.

#ifndef SLATE_CODE_EXECUTOR_H
#define SLATE_CODE_EXECUTOR_H

#include "slate/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slate_exec_result {
    int    compiled;       // syntactically valid (Python: parses)
    int    ran;            // process started
    int    timed_out;
    int    oom;
    int    exit_code;
    char*  stdout_data;    // owned; free with slate_exec_result_free
    size_t stdout_size;
    char*  stderr_data;
    size_t stderr_size;
    double wall_time_ms;
} slate_exec_result_t;

void slate_exec_result_free(slate_exec_result_t* r);

typedef struct slate_exec_limits {
    int timeout_ms;
    int memory_mb;
    int allow_network;       // ignored; always disabled at sandbox level
} slate_exec_limits_t;

typedef struct slate_executor slate_executor_t;

// Linux subprocess + rlimit sandbox. Executes Python code via python3 -c.
// Fork sets RLIMIT_AS (memory), RLIMIT_CPU (wall is enforced from parent).
slate_executor_t* slate_executor_subprocess_new(const char* python_path);
void slate_executor_destroy(slate_executor_t* e);

slate_status_t slate_executor_run(slate_executor_t* e,
                                   const char* code,
                                   const char* stdin_input,
                                   const slate_exec_limits_t* limits,
                                   slate_exec_result_t* out);

#ifdef __cplusplus
}
#endif

#endif
