#!/usr/bin/env python3
"""Build a tiny LLaMA-format GGUF for end-to-end testing of slate_llama_open.

Architecture (chosen small enough for a sandbox + big enough to exercise
multi-head, GQA-capable shape):
    n_layers   = 2
    d_model    = 32   (=> head_dim = 16)
    n_heads    = 2    (MHA: n_kv_heads = 2)
    ffn_hidden = 64
    vocab      = 64
    max_seq    = 128
    theta_base = 10000

All weights f32, deterministic seed.  Tensor naming follows the
llama.cpp convention (token_embd.weight, blk.N.attn_q.weight, ...).
KV metadata follows the `llama.*` namespace.
"""
import struct, sys, numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/slate_tiny_llama.gguf'

# Hyperparameters
N_LAYERS, D, N_HEADS, FFN, VOCAB, MAX_SEQ = 2, 32, 2, 64, 64, 128
HEAD_DIM = D // N_HEADS
N_KV_HEADS = N_HEADS

# GGUF type codes
T_U32, T_STRING, T_F32 = 4, 8, 6
GGML_T_F32 = 0

def w_u32(b, v): b.extend(struct.pack('<I', v))
def w_u64(b, v): b.extend(struct.pack('<Q', v))
def w_f32(b, v): b.extend(struct.pack('<f', v))
def w_str(b, s):
    s = s.encode(); w_u64(b, len(s)); b.extend(s)

def kv_u32(b, k, v):
    w_str(b, k); w_u32(b, T_U32);   w_u32(b, v)
def kv_f32(b, k, v):
    w_str(b, k); w_u32(b, T_F32);   w_f32(b, v)
def kv_str(b, k, v):
    w_str(b, k); w_u32(b, T_STRING); w_str(b, v)

# Generate weights
rng = np.random.default_rng(0xCAFE)
def w(shape):
    return rng.normal(0, 0.02, shape).astype(np.float32)

tensors = []   # list of (name, np_array)

tensors.append(("token_embd.weight",  w((VOCAB, D))))
tensors.append(("output_norm.weight", np.ones((D,), dtype=np.float32)))
tensors.append(("output.weight",      w((VOCAB, D))))

for i in range(N_LAYERS):
    pfx = f"blk.{i}."
    tensors.append((pfx + "attn_norm.weight",   np.ones((D,), dtype=np.float32)))
    # llama.cpp convention: linear weights are stored [out_dim, in_dim].
    tensors.append((pfx + "attn_q.weight",      w((D,                D))))
    tensors.append((pfx + "attn_k.weight",      w((N_KV_HEADS*HEAD_DIM, D))))
    tensors.append((pfx + "attn_v.weight",      w((N_KV_HEADS*HEAD_DIM, D))))
    tensors.append((pfx + "attn_output.weight", w((D,                D))))
    tensors.append((pfx + "ffn_norm.weight",    np.ones((D,), dtype=np.float32)))
    tensors.append((pfx + "ffn_gate.weight",    w((FFN, D))))
    tensors.append((pfx + "ffn_up.weight",      w((FFN, D))))
    tensors.append((pfx + "ffn_down.weight",    w((D,   FFN))))

# Build the GGUF
buf = bytearray(b'GGUF')
w_u32(buf, 3)                    # version
w_u64(buf, len(tensors))         # n_tensors
# n_kv: we'll write the count below; placeholder, count first
KV = []  # (writer_fn,)

# Required kvs
def emit_kvs(buf):
    kv_str(buf, "general.architecture",                "llama")
    kv_u32(buf, "general.alignment",                   32)
    kv_u32(buf, "llama.block_count",                   N_LAYERS)
    kv_u32(buf, "llama.embedding_length",              D)
    kv_u32(buf, "llama.feed_forward_length",           FFN)
    kv_u32(buf, "llama.attention.head_count",          N_HEADS)
    kv_u32(buf, "llama.attention.head_count_kv",       N_KV_HEADS)
    kv_u32(buf, "llama.context_length",                MAX_SEQ)
    kv_u32(buf, "llama.vocab_size",                    VOCAB)
    kv_f32(buf, "llama.rope.freq_base",                10000.0)
    kv_f32(buf, "llama.attention.layer_norm_rms_epsilon", 1e-5)
N_KV = 11
w_u64(buf, N_KV)
emit_kvs(buf)

# Tensor info section
ti_payload = bytearray()   # builds raw tensor data
ti_offsets = []            # offsets within data block

# First compute the offset each tensor will end up at within the data block.
def pad_to(n, align):
    r = n % align
    return n if r == 0 else n + (align - r)

cur = 0
for name, arr in tensors:
    arr_bytes = arr.tobytes()
    cur = pad_to(cur, 32)
    ti_offsets.append(cur)
    ti_payload.extend(b'\0' * (cur - len(ti_payload)))   # pad up
    ti_payload.extend(arr_bytes)
    cur = len(ti_payload)

# Tensor info entries
for (name, arr), off in zip(tensors, ti_offsets):
    w_str(buf, name)
    w_u32(buf, arr.ndim)
    # GGUF stores shape with the SLOWEST-varying dim first (slate's
    # convention also). NumPy `shape` is fastest-last when row-major.
    # llama.cpp shapes are also slowest-first.  For [VOCAB, D] in numpy,
    # llama.cpp expects shape = [D, VOCAB] in its little-endian dump
    # — but slate's gguf reader treats shape[0] as the FIRST dim it
    # reads. Match what slate's existing tests/make_test_gguf.py does:
    # write dims in REVERSED (innermost-first) order.
    for d in arr.shape[::-1]:
        w_u64(buf, d)
    w_u32(buf, GGML_T_F32)
    w_u64(buf, off)

# Pad to alignment
align = 32
while len(buf) % align != 0:
    buf.append(0)
data_offset = len(buf)
buf += ti_payload

with open(OUT, 'wb') as f:
    f.write(buf)

print(f"wrote {OUT}: {len(buf)} bytes, {len(tensors)} tensors, {N_KV} kv entries")
print(f"  arch=llama, n_layers={N_LAYERS}, d_model={D}, n_heads={N_HEADS}, "
        f"head_dim={HEAD_DIM}, ffn={FFN}, vocab={VOCAB}")
