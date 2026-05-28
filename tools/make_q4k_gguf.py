#!/usr/bin/env python3
"""Build a Q4_K_M GGUF for end-to-end testing.

Layout: one tensor "weight" of shape [16, 256] = 16 rows × 1 super-block each.
Round-trip-recovered f32 values are written alongside for the C test to
compare against.
"""
import struct, sys, numpy as np

OUT      = sys.argv[1] if len(sys.argv) > 1 else '/tmp/slate_q4k.gguf'
EXPECTED = '/tmp/slate_q4k_expected.f32'
T_U32, T_STRING = 4, 8
GGML_T_Q4_K = 12

def w_u32(b, v): b.extend(struct.pack('<I', v))
def w_u64(b, v): b.extend(struct.pack('<Q', v))
def w_str(b, s):
    s = s.encode(); w_u64(b, len(s)); b.extend(s)

QK_K = 256

def make_block(weights_256):
    """Quantize 256 f32 to a 144-byte Q4_K block (same algo as make_q4k_test.py)."""
    assert len(weights_256) == QK_K
    w = np.asarray(weights_256, dtype=np.float32).reshape(8, 32)
    sub_scales = np.zeros(8)
    sub_mins   = np.zeros(8)
    q_nibbles  = np.zeros((8, 32), dtype=np.uint8)
    for j in range(8):
        wmin, wmax = w[j].min(), w[j].max()
        if wmax - wmin < 1e-9:
            sub_scales[j] = 1e-9; sub_mins[j] = -wmin; q_nibbles[j] = 0
        else:
            sub_scales[j] = (wmax - wmin) / 15.0
            sub_mins[j]   = -wmin
            q = np.clip(np.round((w[j] - wmin) / sub_scales[j]), 0, 15).astype(np.uint8)
            q_nibbles[j]  = q
    d    = float(sub_scales.max()) / 63.0 if sub_scales.max() > 0 else 1e-9
    dmin = float(sub_mins.max())   / 63.0 if sub_mins.max()   > 0 else 1e-9
    sub_q_scales = np.clip(np.round(sub_scales / d   ), 0, 63).astype(np.uint8)
    sub_q_mins   = np.clip(np.round(sub_mins   / dmin), 0, 63).astype(np.uint8)
    scales_bytes = bytearray(12)
    for j in range(4):
        scales_bytes[j]     = sub_q_scales[j] & 0x3F
        scales_bytes[j + 4] = sub_q_mins[j]   & 0x3F
    for j in range(4, 8):
        scales_bytes[j + 4] = (sub_q_scales[j] & 0x0F) | ((sub_q_mins[j] & 0x0F) << 4)
        scales_bytes[j - 4] |= ((sub_q_scales[j] >> 4) & 0x03) << 6
        scales_bytes[j]     |= ((sub_q_mins[j]   >> 4) & 0x03) << 6
    qs_bytes = bytearray(128)
    for j in range(8):
        for i in range(16):
            qs_bytes[j*16 + i] = (q_nibbles[j, i] & 0x0F) | ((q_nibbles[j, i + 16] & 0x0F) << 4)

    block = bytearray()
    block += np.float16(d).tobytes()
    block += np.float16(dmin).tobytes()
    block += bytes(scales_bytes)
    block += bytes(qs_bytes)
    assert len(block) == 144

    # Recovered f32 (using f16-truncated d, dmin)
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

# 16 rows × 256 weights = 4096 weights total
N_ROWS = 16
np.random.seed(42)
all_weights = np.random.normal(0, 0.5, (N_ROWS, QK_K)).astype(np.float32)
all_blocks   = bytearray()
all_recovered = np.zeros_like(all_weights)
for r in range(N_ROWS):
    block, recon = make_block(all_weights[r])
    all_blocks   += block
    all_recovered[r] = recon

with open(EXPECTED, 'wb') as f:
    f.write(all_recovered.astype(np.float32).tobytes())

# Build the GGUF
buf = bytearray(b'GGUF')
w_u32(buf, 3)            # version
w_u64(buf, 1)            # n_tensors
w_u64(buf, 1)            # n_kv
# One kv: general.alignment = 32
w_str(buf, "general.alignment")
w_u32(buf, T_U32)
w_u32(buf, 32)

# Tensor info: name, n_dims, shape (uint64 each), dtype (u32), offset (u64)
w_str(buf, "weight")
w_u32(buf, 2)            # n_dims
w_u64(buf, QK_K)         # shape[0]  (innermost = 256)
w_u64(buf, N_ROWS)       # shape[1]  (= 16)
w_u32(buf, GGML_T_Q4_K)  # dtype
w_u64(buf, 0)            # offset within data block

# Pad to alignment
align = 32
while len(buf) % align != 0:
    buf.append(0)
data_offset = len(buf)
buf += all_blocks

with open(OUT, 'wb') as f:
    f.write(buf)

print(f"wrote {OUT}: {len(buf)} bytes ({N_ROWS} rows × Q4_K super-block)")
print(f"wrote {EXPECTED}: {N_ROWS * QK_K * 4} bytes (reference recon)")
