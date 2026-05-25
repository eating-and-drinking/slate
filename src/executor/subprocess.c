// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors

#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE
#include "slate/code_executor.h"
#include "slate/error.h"
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

struct slate_executor {
    char* python_path;
};

void slate_exec_result_free(slate_exec_result_t* r) {
    if (!r) return;
    free(r->stdout_data); r->stdout_data = NULL;
    free(r->stderr_data); r->stderr_data = NULL;
}

slate_executor_t* slate_executor_subprocess_new(const char* py) {
    slate_executor_t* e = (slate_executor_t*)calloc(1, sizeof(*e));
    e->python_path = strdup(py ? py : "python3");
    return e;
}
void slate_executor_destroy(slate_executor_t* e) {
    if (!e) return; free(e->python_path); free(e);
}

static double now_ms(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// Read all available bytes from fd into a malloc'd buffer (cap).
static char* slurp_fd(int fd, size_t cap, size_t* out_sz) {
    char* buf = (char*)malloc(cap + 1);
    size_t off = 0;
    while (off < cap) {
        ssize_t n = read(fd, buf + off, cap - off);
        if (n <= 0) break;
        off += (size_t)n;
    }
    buf[off] = 0; *out_sz = off; return buf;
}

slate_status_t slate_executor_run(slate_executor_t* e,
                                   const char* code,
                                   const char* stdin_input,
                                   const slate_exec_limits_t* lim,
                                   slate_exec_result_t* out) {
    if (!e || !code || !out) return SLATE_ERR_INVALID_ARGUMENT;
    memset(out, 0, sizeof(*out));
    out->compiled = 1;  // we don't parse; if python rejects it, ran=1 + nonzero exit_code

    int in_pipe[2], out_pipe[2], err_pipe[2];
    if (pipe(in_pipe) || pipe(out_pipe) || pipe(err_pipe)) return SLATE_ERR_IO;

    double t0 = now_ms();
    pid_t pid = fork();
    if (pid < 0) return SLATE_ERR_IO;

    if (pid == 0) {
        // Child: set rlimits, redirect fds, exec.
        struct rlimit rl;
        if (lim && lim->memory_mb > 0) {
            rl.rlim_cur = rl.rlim_max = (rlim_t)lim->memory_mb * 1024 * 1024;
            setrlimit(RLIMIT_AS, &rl);
        }
        // CPU limit: timeout_ms / 1000 + 1, ceil
        if (lim && lim->timeout_ms > 0) {
            rl.rlim_cur = rl.rlim_max = (rlim_t)((lim->timeout_ms + 999) / 1000 + 1);
            setrlimit(RLIMIT_CPU, &rl);
        }
        // No more than 16 file descriptors
        rl.rlim_cur = rl.rlim_max = 16;
        setrlimit(RLIMIT_NOFILE, &rl);
        // No core dumps
        rl.rlim_cur = rl.rlim_max = 0;
        setrlimit(RLIMIT_CORE, &rl);

        dup2(in_pipe[0], STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);
        dup2(err_pipe[1], STDERR_FILENO);
        close(in_pipe[0]); close(in_pipe[1]);
        close(out_pipe[0]); close(out_pipe[1]);
        close(err_pipe[0]); close(err_pipe[1]);

        execlp(e->python_path, e->python_path, "-c", code, (char*)NULL);
        _exit(127);
    }

    // Parent
    close(in_pipe[0]); close(out_pipe[1]); close(err_pipe[1]);
    if (stdin_input) { write(in_pipe[1], stdin_input, strlen(stdin_input)); }
    close(in_pipe[1]);

    int timeout = (lim && lim->timeout_ms > 0) ? lim->timeout_ms : 5000;
    int killed = 0;
    while (1) {
        int status;
        pid_t r = waitpid(pid, &status, WNOHANG);
        if (r == pid) {
            out->ran = 1;
            if (WIFEXITED(status)) out->exit_code = WEXITSTATUS(status);
            else if (WIFSIGNALED(status)) {
                int sig = WTERMSIG(status);
                if (sig == SIGKILL && killed) out->timed_out = 1;
                if (sig == SIGSEGV) out->oom = 1;  // memory limit often triggers SEGV
                out->exit_code = -sig;
            }
            break;
        } else if (r == -1) {
            if (errno == EINTR) continue;
            break;
        }
        if (now_ms() - t0 > timeout) {
            kill(pid, SIGKILL); killed = 1;
        }
        struct timespec ts = {0, 5 * 1000 * 1000};  // 5ms
        nanosleep(&ts, NULL);
    }
    out->wall_time_ms = now_ms() - t0;

    // Set fds non-blocking before reading stale buffers.
    int flags = fcntl(out_pipe[0], F_GETFL, 0); fcntl(out_pipe[0], F_SETFL, flags | O_NONBLOCK);
    flags = fcntl(err_pipe[0], F_GETFL, 0); fcntl(err_pipe[0], F_SETFL, flags | O_NONBLOCK);
    out->stdout_data = slurp_fd(out_pipe[0], 64 * 1024, &out->stdout_size);
    out->stderr_data = slurp_fd(err_pipe[0], 64 * 1024, &out->stderr_size);
    close(out_pipe[0]); close(err_pipe[0]);
    return SLATE_OK;
}
