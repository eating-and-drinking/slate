// SPDX-License-Identifier: Apache-2.0
#include "slate/slate.h"
#include "slate/code_executor.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    slate_executor_t* e = slate_executor_subprocess_new("python3");
    slate_exec_result_t r;
    slate_exec_limits_t lim = {3000, 128, 0};
    int ok = 1;

    // Test 1: hello world
    slate_executor_run(e, "print('hello sandbox')", NULL, &lim, &r);
    printf("[sb] hello: ran=%d exit=%d wall=%.0fms stdout=%s\n",
           r.ran, r.exit_code, r.wall_time_ms, r.stdout_data);
    ok = ok && r.ran && r.exit_code == 0 && strstr(r.stdout_data, "hello sandbox");
    slate_exec_result_free(&r);

    // Test 2: timeout enforcement (infinite loop)
    slate_executor_run(e, "while True: pass", NULL, &lim, &r);
    printf("[sb] inf loop: ran=%d timed_out=%d wall=%.0fms\n",
           r.ran, r.timed_out, r.wall_time_ms);
    ok = ok && r.timed_out;
    slate_exec_result_free(&r);

    // Test 3: arithmetic
    slate_executor_run(e, "print(2 + 3 * 4)", NULL, &lim, &r);
    printf("[sb] arith: stdout=%s", r.stdout_data);
    ok = ok && r.exit_code == 0 && strstr(r.stdout_data, "14");
    slate_exec_result_free(&r);

    // Test 4: stdin
    slate_executor_run(e, "import sys; print(sum(int(x) for x in sys.stdin.read().split()))",
                        "1 2 3 4 5", &lim, &r);
    printf("[sb] sum stdin: stdout=%s", r.stdout_data);
    ok = ok && r.exit_code == 0 && strstr(r.stdout_data, "15");
    slate_exec_result_free(&r);

    // Test 5: syntax error
    slate_executor_run(e, "this is not valid python", NULL, &lim, &r);
    printf("[sb] syntax err: exit=%d (nonzero expected)\n", r.exit_code);
    ok = ok && r.exit_code != 0;
    slate_exec_result_free(&r);

    slate_executor_destroy(e);
    printf("test_sandbox: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
