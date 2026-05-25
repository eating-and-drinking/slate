#!/usr/bin/env python3
"""Generate a tiny multi-tensor GGUF mimicking LLaMA layer structure."""
import struct, sys, numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/mini_llama.gguf'
N_LAYERS = 4
DIM = 64
T_U32 = 4; T_STRING = 8
GGML_T_Q8_0 = 8

def w_u32(b, v): b.extend(struct.pack('<I', v))
def w_u64(b, v): b.extend(struct.pack('<Q', v))
def w_str(b, s):
    s = s.encode(); w_u64(b, len(s)); b.extend(s)

def f32_to_f16(f):
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

def quantize_q8_0(blk):
    absmax = float(np.max(np.abs(blk)))
    scale = absmax / 127.0 if absmax > 0 else 1.0
    q = np.clip(np.round(blk / scale), -128, 127).astype(np.int8)
    return scale, q

np.random.seed(0xBEEF)
# Header
buf = bytearray(b'GGUF')
w_u32(buf, 3)
w_u64(buf, N_LAYERS)
w_u64(buf, 1)
w_str(buf, 'general.alignment'); w_u32(buf, T_U32); w_u32(buf, 32)

# N_LAYERS tensors, each shape [DIM, DIM] Q8_0
quant_data = bytearray()
offsets = []
for i in range(N_LAYERS):
    name = f'blk.{i}.attn_q.weight'
    w_str(buf, name)
    w_u32(buf, 2)
    w_u64(buf, DIM); w_u64(buf, DIM)
    w_u32(buf, GGML_T_Q8_0)
    offsets.append(len(quant_data))
    w_u64(buf, offsets[-1])
    # quantize
    W = np.random.uniform(-0.1, 0.1, (DIM, DIM)).astype(np.float32)
    for r in range(DIM):
        for blk_start in range(0, DIM, 32):
            blk = W[r, blk_start:blk_start+32]
            scale, q = quantize_q8_0(blk)
            quant_data.extend(struct.pack('<H', f32_to_f16(scale)))
            quant_data.extend(q.tobytes())

# Pad before data section
pad = (32 - (len(buf) % 32)) % 32
buf.extend(b'\x00' * pad)
buf.extend(quant_data)
with open(OUT, 'wb') as f: f.write(buf)
print(f'wrote {OUT}, {len(buf)} bytes, {N_LAYERS} tensors, each {DIM}x{DIM} Q8_0')
