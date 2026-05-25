// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// quant.c — GGML Q8_0 / Q4_0 dequantization.

#include "slate/quant.h"
#include "slate/precision.h"
#include <stdint.h>
#include <string.h>

void slate_dequant_q8_0(float* dst, const void* src, int64_t n) {
    const uint8_t* p = (const uint8_t*)src;
    int64_t blocks = n / SLATE_QUANT_BLOCK_ELEMS;
    for (int64_t b = 0; b < blocks; ++b) {
        const uint8_t* blk = p + b * SLATE_Q8_0_BLOCK_SIZE;
        uint16_t scale16; memcpy(&scale16, blk, 2);
        float d = slate_f16_to_f32(scale16);
        const int8_t* q = (const int8_t*)(blk + 2);
        float* o = dst + b * SLATE_QUANT_BLOCK_ELEMS;
        for (int i = 0; i < SLATE_QUANT_BLOCK_ELEMS; ++i) {
            o[i] = d * (float)q[i];
        }
    }
}

void slate_dequant_q4_0(float* dst, const void* src, int64_t n) {
    const uint8_t* p = (const uint8_t*)src;
    int64_t blocks = n / SLATE_QUANT_BLOCK_ELEMS;
    for (int64_t b = 0; b < blocks; ++b) {
        const uint8_t* blk = p + b * SLATE_Q4_0_BLOCK_SIZE;
        uint16_t scale16; memcpy(&scale16, blk, 2);
        float d = slate_f16_to_f32(scale16);
        const uint8_t* q = blk + 2;
        float* o = dst + b * SLATE_QUANT_BLOCK_ELEMS;
        // 16 bytes encode 32 4-bit values. ggml convention: low nibbles are
        // the first 16 elements, high nibbles are the last 16 elements.
        for (int i = 0; i < 16; ++i) {
            uint8_t v = q[i];
            int lo = (int)(v & 0x0f) - 8;
            int hi = (int)(v >> 4)   - 8;
            o[i]      = d * (float)lo;
            o[i + 16] = d * (float)hi;
        }
    }
}
