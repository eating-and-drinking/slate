#define _POSIX_C_SOURCE 200809L
#include "slate/process_reward.h"
#include "slate/http_teacher.h"
#include <stdio.h>
#include <string.h>

// Simple step scorer: +1 if the step contains "good", -1 if "bad", 0 otherwise.
static float score_step(const char* text, int idx, void* ud) {
    (void)idx; (void)ud;
    if (strstr(text, "good")) return 1.0f;
    if (strstr(text, "bad")) return -1.0f;
    return 0.0f;
}

// Mock transport: just echoes the request body as the response.
static int mock_transport(const char* ep, const char* body, char** out, void* ud) {
    (void)ep; (void)ud;
    *out = strdup(body); return 0;
}

int main(void) {
    int ok = 1;
    // PRM
    const char* response = "step1: bad approach. step2: but actually this is good. step3: also good.";
    float sum = slate_process_reward(response, ".", score_step, NULL, SLATE_PRM_SUM);
    float mean = slate_process_reward(response, ".", score_step, NULL, SLATE_PRM_MEAN);
    float mn  = slate_process_reward(response, ".", score_step, NULL, SLATE_PRM_MIN);
    printf("[prm] sum=%.2f mean=%.2f min=%.2f\n", sum, mean, mn);
    ok = ok && sum == 1.0f && mn == -1.0f;

    // HTTP teacher with mock transport
    slate_http_teacher_t* t = slate_http_teacher_openai_new("gpt-4o-mini", "fake-key");
    // Without transport: error
    slate_http_response_t r = slate_http_teacher_generate(t, "hi", 10);
    printf("[http] no transport: %s\n", r.error ? r.error : "(no error)");
    ok = ok && r.error != NULL;
    slate_http_response_free(&r);
    // With transport
    slate_http_teacher_set_transport(t, mock_transport, NULL);
    r = slate_http_teacher_generate(t, "hello world", 8);
    printf("[http] mock transport: text=%s\n", r.text ? r.text : "(null)");
    ok = ok && r.text != NULL && strstr(r.text, "hello world");
    slate_http_response_free(&r);
    slate_http_teacher_destroy(t);
    printf("test_prm_http: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
