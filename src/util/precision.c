// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 The Slate Authors
//
// precision.c — f16/bf16 conversion implementations.

#include "slate/precision.h"
#include <stdint.h>
#include <string.h>

float slate_f16_to_f32(uint16_t h) {
    // Branchless conversion via bit manipulation.
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp16 = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    if (exp16 == 0) {
        if (mant == 0) { uint32_t v = sign; float f; memcpy(&f, &v, 4); return f; }
        // Subnormal: shift mantissa up until the implicit 1 is exposed.
        int e = -1;
        do { mant <<= 1; e++; } while ((mant & 0x400u) == 0);
        mant &= 0x3ffu;
        uint32_t v = sign | ((127u - 15u - (uint32_t)e) << 23) | (mant << 13);
        float f; memcpy(&f, &v, 4); return f;
    } else if (exp16 == 31) {
        uint32_t v = sign | 0x7f800000u | (mant << 13);
        float f; memcpy(&f, &v, 4); return f;
    }
    uint32_t v = sign | ((exp16 + 127u - 15u) << 23) | (mant << 13);
    float f; memcpy(&f, &v, 4); return f;
}

uint16_t slate_f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp32 = (int32_t)((x >> 23) & 0xffu) - 127;
    uint32_t mant = x & 0x7fffffu;
    if (exp32 == 128) {
        // Inf / NaN
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x200 : 0));
    }
    if (exp32 > 15) return (uint16_t)(sign | 0x7c00u);  // overflow -> inf
    if (exp32 < -14) {
        // Subnormal or underflow.
        if (exp32 < -25) return (uint16_t)sign;
        mant = (mant | 0x800000u) >> (uint32_t)(-14 - exp32 + 13);
        return (uint16_t)(sign | mant);
    }
    return (uint16_t)(sign | ((uint32_t)(exp32 + 15) << 10) | (mant >> 13));
}

float slate_bf16_to_f32(uint16_t b) {
    uint32_t v = (uint32_t)b << 16;
    float f; memcpy(&f, &v, 4); return f;
}

uint16_t slate_f32_to_bf16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    // Round-to-nearest-even.
    uint32_t rounded = x + ((x >> 16) & 1u) + 0x7fffu;
    return (uint16_t)(rounded >> 16);
}

void slate_f16_to_f32_n(float* d, const uint16_t* s, size_t n) {
    for (size_t i = 0; i < n; ++i) d[i] = slate_f16_to_f32(s[i]);
}
void slate_f32_to_f16_n(uint16_t* d, const float* s, size_t n) {
    for (size_t i = 0; i < n; ++i) d[i] = slate_f32_to_f16(s[i]);
}
void slate_bf16_to_f32_n(float* d, const uint16_t* s, size_t n) {
    for (size_t i = 0; i < n; ++i) d[i] = slate_bf16_to_f32(s[i]);
}
void slate_f32_to_bf16_n(uint16_t* d, const float* s, size_t n) {
    for (size_t i = 0; i < n; ++i) d[i] = slate_f32_to_bf16(s[i]);
}
