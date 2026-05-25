// SPDX-License-Identifier: Apache-2.0
#include "slate/precision.h"
#include "slate/quant.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    int ok = 1;
    // f16 round-trip on a known value
    float v = 1.5f;
    uint16_t h = slate_f32_to_f16(v);
    float back = slate_f16_to_f32(h);
    printf("[prec] f16: 1.5 -> 0x%04x -> %.6f\n", h, back);
    ok = ok && fabsf(back - v) < 1e-3f;

    // bf16 round-trip
    v = 3.14159f;
    uint16_t bb = slate_f32_to_bf16(v);
    float back2 = slate_bf16_to_f32(bb);
    printf("[prec] bf16: 3.14159 -> 0x%04x -> %.6f (lossy)\n", bb, back2);
    ok = ok && fabsf(back2 - v) < 0.02f;

    // Q8_0 round-trip: build a synthetic block then dequant
    uint8_t block[SLATE_Q8_0_BLOCK_SIZE];
    uint16_t scale = slate_f32_to_f16(0.1f);
    memcpy(block, &scale, 2);
    for (int i = 0; i < 32; ++i) ((int8_t*)(block + 2))[i] = (int8_t)(i - 16);  // -16..+15
    float out8[32];
    slate_dequant_q8_0(out8, block, 32);
    printf("[q8_0] dequant first 4: %.3f %.3f %.3f %.3f (expected -1.6, -1.5, -1.4, -1.3)\n",
           out8[0], out8[1], out8[2], out8[3]);
    // Check: out[i] = 0.1 * (i - 16)
    int q8_ok = 1;
    for (int i = 0; i < 32; ++i) {
        float expect = 0.1f * (i - 16);
        if (fabsf(out8[i] - expect) > 0.01f) { q8_ok = 0; break; }
    }
    printf("[q8_0] all 32 values match: %s\n", q8_ok ? "yes" : "no");
    ok = ok && q8_ok;

    // Q4_0 round-trip
    uint8_t block4[SLATE_Q4_0_BLOCK_SIZE];
    scale = slate_f32_to_f16(0.5f);
    memcpy(block4, &scale, 2);
    // Set nibbles: pos i (low) = i, pos i+16 (high) = 15 - i
    for (int i = 0; i < 16; ++i) block4[2 + i] = (uint8_t)(i | ((15 - i) << 4));
    float out4[32];
    slate_dequant_q4_0(out4, block4, 32);
    printf("[q4_0] dequant first 4: %.3f %.3f %.3f %.3f\n", out4[0], out4[1], out4[2], out4[3]);
    printf("[q4_0] last 4: %.3f %.3f %.3f %.3f\n", out4[28], out4[29], out4[30], out4[31]);
    // out[i]      = 0.5 * (i - 8)        for i in [0, 16)
    // out[i+16]   = 0.5 * ((15-i) - 8) = 0.5 * (7-i)
    int q4_ok = 1;
    for (int i = 0; i < 16; ++i) {
        float e1 = 0.5f * (float)(i - 8);
        float e2 = 0.5f * (float)((15 - i) - 8);
        if (fabsf(out4[i] - e1) > 0.01f) { q4_ok = 0; break; }
        if (fabsf(out4[i + 16] - e2) > 0.01f) { q4_ok = 0; break; }
    }
    printf("[q4_0] all 32 values match: %s\n", q4_ok ? "yes" : "no");
    ok = ok && q4_ok;

    printf("test_quant: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
