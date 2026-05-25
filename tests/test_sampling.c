// SPDX-License-Identifier: Apache-2.0
#include "slate/sampling.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    // 4-token vocab with one clearly dominant logit.
    float logits[4] = {0.1f, 2.5f, 0.2f, -1.0f};
    uint64_t rng = 0xC0FFEE;

    // greedy (T=0) should always pick token 1.
    slate_sampler_config_t g = {0.0f, 0, 0.0f, 0};
    for (int i = 0; i < 10; ++i) {
        int t = slate_sample_token(logits, 4, &g, &rng);
        if (t != 1) { puts("greedy FAIL"); return 1; }
    }
    puts("greedy OK");

    // temperature 1.0, no filter: histogram should approximate softmax.
    slate_sampler_config_t t = {1.0f, 0, 0.0f, 0};
    int counts[4] = {0};
    for (int i = 0; i < 10000; ++i) counts[slate_sample_token(logits, 4, &t, &rng)]++;
    printf("T=1.0 sample counts: %d %d %d %d\n", counts[0], counts[1], counts[2], counts[3]);
    // Token 1 should dominate.
    int ok = counts[1] > counts[0] && counts[1] > counts[2] && counts[1] > counts[3];

    // top_k=1 with T=1.0: forced to argmax.
    slate_sampler_config_t k1 = {1.0f, 1, 0.0f, 0};
    int saw_only_1 = 1;
    for (int i = 0; i < 100; ++i) {
        if (slate_sample_token(logits, 4, &k1, &rng) != 1) { saw_only_1 = 0; break; }
    }
    printf("top_k=1: always token 1: %s\n", saw_only_1 ? "yes" : "no");
    ok = ok && saw_only_1;

    printf("test_sampling: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
