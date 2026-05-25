#!/usr/bin/env python3
"""Build a Q8_0 GGUF for end-to-end testing."""
import struct, sys, numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/slate_q8.gguf'
T_U32, T_STRING = 4, 8
GGML_T_Q8_0 = 8

def w_u32(b, v): b.extend(struct.pack('<I', v))
def w_u64(b, v): b.extend(struct.pack('<Q', v))
def w_str(b, s):
    s = s.encode(); w_u64(b, len(s)); b.extend(s)

# We'll store a [32, 4] = 128-element matrix quantized to Q8_0 (4 blocks of 32).
# Each row of the original matrix is a different scale and pattern.

def f32_to_f16(f):
    # Same algorithm as our C code
    x = struct.unpack('<I', struct.pack('<f', f))[0]
    sign = (x >> 16) & 0x8000
    exp = ((x >> 23) & 0xff) - 127
    mant = x & 0x7fffff
    if exp == 128: return sign | 0x7c00 | (0x200 if mant else 0)
    if exp > 15: return sign | 0x7c00
    if exp < -14:
        if exp < -25: return sign
        mant = (mant | 0x800000) >> (-14 - exp + 13)
        return sign | mant
    return sign | ((exp + 15) << 10) | (mant >> 13)

# Original floats: 128 values, distinct patterns per 32-block
np.random.seed(42)
orig = np.random.uniform(-1, 1, 128).astype(np.float32)

# Quantize per block: scale = max(|x|) / 127, q = round(x / scale)
quant_blocks = bytearray()
recovered = np.zeros_like(orig)
for b in range(4):
    blk = orig[b*32:(b+1)*32]
    absmax = float(np.max(np.abs(blk)))
    scale = absmax / 127.0 if absmax > 0 else 1.0
    q = np.clip(np.round(blk / scale), -128, 127).astype(np.int8)
    quant_blocks.extend(struct.pack('<H', f32_to_f16(scale)))
    quant_blocks.extend(q.tobytes())
    recovered[b*32:(b+1)*32] = scale * q.astype(np.float32)

# Save the "expected dequant" alongside the GGUF as a sibling .npy
recovered.tofile('/tmp/slate_q8_expected.f32')

buf = bytearray(b'GGUF')
w_u32(buf, 3)
w_u64(buf, 1)   # n_tensors
w_u64(buf, 1)   # n_kv
w_str(buf, 'general.alignment'); w_u32(buf, T_U32); w_u32(buf, 32)
w_str(buf, 'quant.weight')
w_u32(buf, 2); w_u64(buf, 32); w_u64(buf, 4)
w_u32(buf, GGML_T_Q8_0)
w_u64(buf, 0)
pad = (32 - (len(buf) % 32)) % 32
buf.extend(b'\x00' * pad)
buf.extend(quant_blocks)
with open(OUT, 'wb') as f: f.write(buf)
print(f"wrote {OUT}, {len(buf)} bytes, {len(quant_blocks)} bytes quantized data")
print(f"orig[0:4]      = {orig[0:4]}")
print(f"recovered[0:4] = {recovered[0:4]}")
print(f"max abs error  = {np.max(np.abs(orig - recovered)):.4f}")
