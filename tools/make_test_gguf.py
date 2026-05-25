#!/usr/bin/env python3
"""Generate a tiny synthetic GGUF file for round-trip testing."""
import struct, sys, os

OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/slate_test.gguf'

# GGUF type tags
T_U32 = 4
T_STRING = 8
GGML_T_F32 = 0

def w_u32(b, v): b.extend(struct.pack('<I', v))
def w_u64(b, v): b.extend(struct.pack('<Q', v))
def w_str(b, s):
    s = s.encode('utf-8'); w_u64(b, len(s)); b.extend(s)
def w_f32(b, v): b.extend(struct.pack('<f', v))

buf = bytearray()
buf.extend(b'GGUF')          # magic
w_u32(buf, 3)                # version

# We'll have 2 tensors and 2 kv entries.
n_tensors = 2
n_kv = 2
w_u64(buf, n_tensors)
w_u64(buf, n_kv)

# kv 1: general.architecture = "llama"
w_str(buf, "general.architecture")
w_u32(buf, T_STRING)
w_str(buf, "llama")
# kv 2: general.alignment = 32
w_str(buf, "general.alignment")
w_u32(buf, T_U32)
w_u32(buf, 32)

# Tensor 1: shape [4,3] of F32
w_str(buf, "test.weight_a")
w_u32(buf, 2)                # n_dims
w_u64(buf, 4); w_u64(buf, 3)  # shape (gguf stores in reverse? actually direct order)
w_u32(buf, GGML_T_F32)
w_u64(buf, 0)                # offset 0 in data section

# Tensor 2: shape [3] of F32
w_str(buf, "test.bias_b")
w_u32(buf, 1)
w_u64(buf, 3)
w_u32(buf, GGML_T_F32)
w_u64(buf, 4 * 3 * 4)         # offset = sizeof tensor1 data

# Pad to alignment 32
pad = (32 - (len(buf) % 32)) % 32
buf.extend(b'\x00' * pad)

# Data section
# Tensor 1: 12 floats = 0.1, 0.2, ..., 1.2
for i in range(12): w_f32(buf, 0.1 * (i + 1))
# Tensor 2: 3 floats = 10, 20, 30
for v in [10.0, 20.0, 30.0]: w_f32(buf, v)

with open(OUT, 'wb') as f: f.write(buf)
print(f"wrote {OUT}, {len(buf)} bytes")
