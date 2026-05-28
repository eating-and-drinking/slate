#!/usr/bin/env python3
"""Build a RoPE reference for end-to-end testing in C.

Emits /tmp/slate_rope_ref.bin containing:
    int32  B, T, n_heads, head_dim
    float  theta_base
    float  x_in [B * T * n_heads * head_dim]
    int32  positions [T]
    float  y_ref [B * T * n_heads * head_dim]

The C test reads these and verifies slate_op_rope(x_in, positions, theta_base)
matches y_ref within fp32 tolerance.

GGML "non-interleaved" convention:
    y[..., i]        =  x[..., i] * cos - x[..., i + hd/2] * sin
    y[..., i + hd/2] =  x[..., i] * sin + x[..., i + hd/2] * cos
with cos/sin determined by angle = position * theta_base^(-2i / hd)
for i in [0, hd/2).
"""
import struct, numpy as np

B, T, n_heads, head_dim = 2, 5, 4, 16
theta_base = 10000.0

rng = np.random.default_rng(0xABCD)
x = rng.normal(0, 1, (B, T, n_heads, head_dim)).astype(np.float32)
positions = np.arange(T, dtype=np.int32) + 7   # start at position 7 just for variety

half = head_dim // 2
i_idx = np.arange(half)
inv_freq = theta_base ** (-2.0 * i_idx / head_dim)  # [half]

# angles: position[t] * inv_freq[i] -> shape [T, half]
angles = positions[:, None].astype(np.float32) * inv_freq.astype(np.float32)
cos = np.cos(angles)  # [T, half]
sin = np.sin(angles)

y = np.zeros_like(x)
for b in range(B):
    for t in range(T):
        for h in range(n_heads):
            a = x[b, t, h, :half]
            d = x[b, t, h, half:]
            y[b, t, h, :half] = a * cos[t] - d * sin[t]
            y[b, t, h, half:] = a * sin[t] + d * cos[t]

with open('/tmp/slate_rope_ref.bin', 'wb') as f:
    f.write(struct.pack('<iiii', B, T, n_heads, head_dim))
    f.write(struct.pack('<f', theta_base))
    f.write(x.astype(np.float32).tobytes())
    f.write(positions.astype(np.int32).tobytes())
    f.write(y.astype(np.float32).tobytes())

n = B * T * n_heads * head_dim
print(f"wrote /tmp/slate_rope_ref.bin: B={B} T={T} H={n_heads} hd={head_dim}, {n} elements")
