#!/usr/bin/env python3
# Build a Q4_K_M reference block and emit:
#   - the raw 144-byte block bytes
#   - the original f32 weights (for round-trip comparison)
# Used by tests/test_q4k.c to verify slate's dequant matches the spec.

import struct, math, numpy as np

QK_K = 256

def make_block(weights_256):
    """Quantize 256 f32 weights into one Q4_K block (144 bytes)."""
    assert len(weights_256) == QK_K
    w = np.asarray(weights_256, dtype=np.float32).reshape(8, 32)

    # Per-sub-block: choose scale, min so that
    #   w_i ≈ scale * q_i - min  with q_i in [0, 15], scale, min ≥ 0
    # i.e. q_i = clip(round((w_i + min) / scale), 0, 15).
    # We pick min = -w_min (so the smallest weight maps to q=0), and
    # scale = (w_max - w_min) / 15.
    sub_scales = np.zeros(8)
    sub_mins   = np.zeros(8)
    q_nibbles  = np.zeros((8, 32), dtype=np.uint8)
    for j in range(8):
        wmin = w[j].min()
        wmax = w[j].max()
        if wmax - wmin < 1e-9:
            sub_scales[j] = 1e-9
            sub_mins[j]   = -wmin
            q_nibbles[j]  = 0
        else:
            sub_scales[j] = (wmax - wmin) / 15.0
            sub_mins[j]   = -wmin
            q = np.round((w[j] - wmin) / sub_scales[j]).astype(np.int32)
            q = np.clip(q, 0, 15).astype(np.uint8)
            q_nibbles[j] = q
    # Pack: super-block d, dmin such that
    #   scale_j_actual ≈ d * sub_q_scale_j   where sub_q_scale_j ∈ [0, 63]
    #   min_j_actual   ≈ dmin * sub_q_min_j  where sub_q_min_j   ∈ [0, 63]
    d    = float(sub_scales.max()) / 63.0 if sub_scales.max() > 0 else 1e-9
    dmin = float(sub_mins.max())   / 63.0 if sub_mins.max()   > 0 else 1e-9
    sub_q_scales = np.clip(np.round(sub_scales / d   ), 0, 63).astype(np.uint8)
    sub_q_mins   = np.clip(np.round(sub_mins   / dmin), 0, 63).astype(np.uint8)

    # Pack the 12-byte scales[] array following GGML convention:
    #   for j < 4: low 6 bits in scales[j] (scale) and scales[j+4] (min)
    #   for j ≥ 4: scales[j+4] holds low-4-of-scale | low-4-of-min<<4
    #              upper 2 bits of scale_j come from scales[j-4]>>6
    #              upper 2 bits of min_j   come from scales[j  ]>>6
    scales_bytes = bytearray(12)
    for j in range(4):
        scales_bytes[j]     = sub_q_scales[j] & 0x3F
        scales_bytes[j + 4] = sub_q_mins[j]   & 0x3F
    for j in range(4, 8):
        # Pack low 4 of scale and min into byte j+4
        scales_bytes[j + 4] = (sub_q_scales[j] & 0x0F) | ((sub_q_mins[j] & 0x0F) << 4)
        # Pack high 2 bits into upper 2 bits of bytes j-4 (scale) and j (min)
        scales_bytes[j - 4] |= ((sub_q_scales[j] >> 4) & 0x03) << 6
        scales_bytes[j]     |= ((sub_q_mins[j]   >> 4) & 0x03) << 6

    # Pack 256 nibbles into 128 bytes: per sub-block, qs[j*16+i] = lo | hi<<4
    qs_bytes = bytearray(128)
    for j in range(8):
        for i in range(16):
            qs_bytes[j*16 + i] = (q_nibbles[j, i] & 0x0F) | ((q_nibbles[j, i + 16] & 0x0F) << 4)

    # Write block: f16 d, f16 dmin, 12 bytes scales, 128 bytes qs
    block = bytearray()
    block += np.float16(d).tobytes()
    block += np.float16(dmin).tobytes()
    block += bytes(scales_bytes)
    block += bytes(qs_bytes)
    assert len(block) == 144

    # Round-trip reconstruction so the test knows what to expect
    reconstructed = np.zeros((8, 32), dtype=np.float32)
    d_real    = float(np.float16(d))
    dmin_real = float(np.float16(dmin))
    for j in range(8):
        sub_d = d_real * sub_q_scales[j]
        sub_m = dmin_real * sub_q_mins[j]
        for i in range(16):
            lo = (qs_bytes[j*16+i] & 0x0F)
            hi = (qs_bytes[j*16+i] >> 4)
            reconstructed[j, i]      = sub_d * lo - sub_m
            reconstructed[j, i + 16] = sub_d * hi - sub_m
    return bytes(block), reconstructed.reshape(-1)

# Generate a single test super-block: a sinusoid offset around zero
rng = np.random.default_rng(42)
weights = rng.normal(0, 1, QK_K).astype(np.float32) * 0.5
block, recon = make_block(weights)

# Write to /tmp for the C test to read
with open("/tmp/slate_q4k_test.bin", "wb") as f:
    f.write(block)
    # also write the 256 f32 originals + reconstruction so C can compare
    f.write(weights.astype(np.float32).tobytes())
    f.write(recon.astype(np.float32).tobytes())

print(f"wrote /tmp/slate_q4k_test.bin: {len(block)} bytes block + 256*4 orig + 256*4 recon")
err = np.abs(weights - recon).mean()
print(f"mean dequant error vs original f32: {err:.6f}")
