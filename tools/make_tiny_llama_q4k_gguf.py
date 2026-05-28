#!/usr/bin/env python3
"""Build a Q4_K LLaMA-format GGUF — projection weights are Q4_K, norms +
embedding stay f32.  Mirrors llama.cpp's typical Q4_K_M deployment
configuration.  The reference numpy forward dequantises the Q4_K weights
to f32 before computing so the C test can compare against bit-equivalent
output.

Choose dims so every linear weight's K dimension is a multiple of 256
(Q4_K super-block size).
"""
import struct, numpy as np

OUT_GGUF = '/tmp/slate_tiny_llama_q4k.gguf'
OUT_REF  = '/tmp/slate_tiny_llama_q4k_ref.bin'

# Need K (the "in" dim of every linear) to be ≥ 256 for Q4_K super-blocks.
N_LAYERS, D, N_HEADS, FFN, VOCAB, MAX_SEQ = 2, 256, 4, 512, 64, 128
HEAD_DIM = D // N_HEADS
N_KV_HEADS = N_HEADS
RMS_EPS = 1e-5
THETA_BASE = 10000.0
QK_K = 256

# GGUF type codes
T_U32, T_STRING, T_F32 = 4, 8, 6
GGML_T_F32 = 0
GGML_T_Q4_K = 12

def w_u32(b, v): b.extend(struct.pack('<I', v))
def w_u64(b, v): b.extend(struct.pack('<Q', v))
def w_f32(b, v): b.extend(struct.pack('<f', v))
def w_str(b, s):
    s = s.encode(); w_u64(b, len(s)); b.extend(s)
def kv_u32(b, k, v): w_str(b, k); w_u32(b, T_U32); w_u32(b, v)
def kv_f32(b, k, v): w_str(b, k); w_u32(b, T_F32); w_f32(b, v)
def kv_str(b, k, v): w_str(b, k); w_u32(b, T_STRING); w_str(b, v)

# Q4_K quantisation (reuses make_q4k_test.py's algorithm)
def quant_block_q4k(weights_256):
    assert len(weights_256) == QK_K
    w = np.asarray(weights_256, dtype=np.float32).reshape(8, 32)
    sub_scales = np.zeros(8); sub_mins = np.zeros(8)
    q_nibbles = np.zeros((8, 32), dtype=np.uint8)
    for j in range(8):
        wmin, wmax = w[j].min(), w[j].max()
        if wmax - wmin < 1e-9:
            sub_scales[j] = 1e-9; sub_mins[j] = -wmin; q_nibbles[j] = 0
        else:
            sub_scales[j] = (wmax - wmin) / 15.0
            sub_mins[j]   = -wmin
            q = np.clip(np.round((w[j] - wmin) / sub_scales[j]), 0, 15).astype(np.uint8)
            q_nibbles[j] = q
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
    # Recovered f32 for the reference forward pass
    rec = np.zeros((8, 32), dtype=np.float32)
    d_real    = float(np.float16(d))
    dmin_real = float(np.float16(dmin))
    for j in range(8):
        sub_d = d_real * sub_q_scales[j]
        sub_m = dmin_real * sub_q_mins[j]
        for i in range(16):
            lo = (qs_bytes[j*16+i] & 0x0F)
            hi = (qs_bytes[j*16+i] >> 4)
            rec[j, i]      = sub_d * lo - sub_m
            rec[j, i + 16] = sub_d * hi - sub_m
    return bytes(block), rec.reshape(-1)

def quant_q4k_tensor(arr):
    """arr is (M, K) row-major f32. Returns (Q4_K bytes, recovered f32)."""
    assert arr.dtype == np.float32 and arr.ndim == 2
    M, K = arr.shape
    assert K % QK_K == 0
    blocks = bytearray()
    rec = np.zeros_like(arr)
    for m in range(M):
        row = arr[m]
        for b in range(K // QK_K):
            bb, rr = quant_block_q4k(row[b*QK_K:(b+1)*QK_K])
            blocks += bb
            rec[m, b*QK_K:(b+1)*QK_K] = rr
    return bytes(blocks), rec

rng = np.random.default_rng(0xCAFE)
def w(shape):
    return rng.normal(0, 0.02, shape).astype(np.float32)

# Build weights and their Q4_K-roundtripped versions.
weights = {}   # name -> (raw_f32, q4k_bytes_or_None, dequantised_f32)
def add_f32(name, arr):
    weights[name] = (arr, None, arr)
def add_q4k(name, arr):
    qbytes, rec = quant_q4k_tensor(arr)
    weights[name] = (arr, qbytes, rec)

add_f32("token_embd",  w((VOCAB, D)))
add_f32("output_norm", np.ones((D,), dtype=np.float32))
add_f32("output",      w((VOCAB, D)))   # keep f32 for simplicity

for i in range(N_LAYERS):
    pfx = f"L{i}."
    add_f32(pfx + "attn_norm", np.ones((D,), dtype=np.float32))
    add_q4k(pfx + "Wq",        w((D, D)))
    add_q4k(pfx + "Wk",        w((N_KV_HEADS*HEAD_DIM, D)))
    add_q4k(pfx + "Wv",        w((N_KV_HEADS*HEAD_DIM, D)))
    add_q4k(pfx + "Wo",        w((D, D)))
    add_f32(pfx + "ffn_norm",  np.ones((D,), dtype=np.float32))
    add_q4k(pfx + "ffn_gate",  w((FFN, D)))
    add_q4k(pfx + "ffn_up",    w((FFN, D)))
    add_q4k(pfx + "ffn_down",  w((D, FFN)))

# ----- Build the GGUF -----
buf = bytearray(b'GGUF')
w_u32(buf, 3)
n_tensors = 3 + N_LAYERS * 9   # global 3 + 9 per block
w_u64(buf, n_tensors)
N_KV = 11
w_u64(buf, N_KV)

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

# Map slate-side name to GGUF name
gguf_name = {
    "token_embd":  "token_embd.weight",
    "output_norm": "output_norm.weight",
    "output":      "output.weight",
}
for i in range(N_LAYERS):
    pfx_src = f"L{i}."
    pfx_g   = f"blk.{i}."
    gguf_name[pfx_src + "attn_norm"] = pfx_g + "attn_norm.weight"
    gguf_name[pfx_src + "Wq"]        = pfx_g + "attn_q.weight"
    gguf_name[pfx_src + "Wk"]        = pfx_g + "attn_k.weight"
    gguf_name[pfx_src + "Wv"]        = pfx_g + "attn_v.weight"
    gguf_name[pfx_src + "Wo"]        = pfx_g + "attn_output.weight"
    gguf_name[pfx_src + "ffn_norm"]  = pfx_g + "ffn_norm.weight"
    gguf_name[pfx_src + "ffn_gate"]  = pfx_g + "ffn_gate.weight"
    gguf_name[pfx_src + "ffn_up"]    = pfx_g + "ffn_up.weight"
    gguf_name[pfx_src + "ffn_down"]  = pfx_g + "ffn_down.weight"

# Compute per-tensor offsets in data section
ti_payload = bytearray()
ti_offsets = []
def pad_to(n, align):
    r = n % align
    return n if r == 0 else n + (align - r)
cur = 0
ordered = list(weights.items())
for name, (raw, qbytes, rec) in ordered:
    payload = qbytes if qbytes is not None else raw.tobytes()
    cur = pad_to(cur, 32)
    ti_offsets.append(cur)
    ti_payload.extend(b'\0' * (cur - len(ti_payload)))
    ti_payload.extend(payload)
    cur = len(ti_payload)

for (name, (raw, qbytes, rec)), off in zip(ordered, ti_offsets):
    gn = gguf_name[name]
    w_str(buf, gn)
    shape = raw.shape
    w_u32(buf, len(shape))
    for d in shape[::-1]:
        w_u64(buf, d)
    w_u32(buf, GGML_T_Q4_K if qbytes is not None else GGML_T_F32)
    w_u64(buf, off)

# Pad to alignment
while len(buf) % 32 != 0:
    buf.append(0)
buf += ti_payload

with open(OUT_GGUF, 'wb') as f: f.write(buf)
print(f"wrote {OUT_GGUF}: {len(buf)} bytes")

# ----- Numpy reference forward (using DEQUANTISED weights) -----
def rmsnorm(x, ws, eps):
    rms = np.sqrt((x*x).mean(axis=-1, keepdims=True) + eps)
    return (x/rms) * ws
def silu(x): return x / (1.0 + np.exp(-x))
def rope(x, pos, theta, hd):
    half = hd // 2
    inv = theta ** (-2.0 * np.arange(half) / hd)
    a = pos * inv
    c, s = np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)
    x = x.copy()
    lo = x[..., :half].copy(); hi = x[..., half:].copy()
    x[..., :half] = lo*c - hi*s
    x[..., half:] = lo*s + hi*c
    return x

# Use the DEQUANTISED weights for the reference forward pass.
W = {k: rec for k, (_, _, rec) in weights.items()}

prompt = np.array([1, 7, 13, 25, 3], dtype=np.int32)
K_cache = np.zeros((N_LAYERS, MAX_SEQ, N_KV_HEADS*HEAD_DIM), dtype=np.float32)
V_cache = np.zeros((N_LAYERS, MAX_SEQ, N_KV_HEADS*HEAD_DIM), dtype=np.float32)
last = None
for pi, tok in enumerate(prompt):
    x = W["token_embd"][tok].copy()
    for li in range(N_LAYERS):
        h = rmsnorm(x, W[f"L{li}.attn_norm"], RMS_EPS)
        q = W[f"L{li}.Wq"] @ h
        k = W[f"L{li}.Wk"] @ h
        v = W[f"L{li}.Wv"] @ h
        q_h = q.reshape(N_HEADS, HEAD_DIM); q_h = rope(q_h, pi, THETA_BASE, HEAD_DIM)
        k_h = k.reshape(N_KV_HEADS, HEAD_DIM); k_h = rope(k_h, pi, THETA_BASE, HEAD_DIM)
        K_cache[li, pi] = k_h.reshape(-1); V_cache[li, pi] = v
        L = pi + 1
        out = np.zeros((N_HEADS, HEAD_DIM), dtype=np.float32)
        for hi in range(N_HEADS):
            Kh = K_cache[li, :L].reshape(L, N_KV_HEADS, HEAD_DIM)[:, hi]
            Vh = V_cache[li, :L].reshape(L, N_KV_HEADS, HEAD_DIM)[:, hi]
            sc = (Kh @ q_h[hi]) / np.sqrt(HEAD_DIM)
            sc = np.exp(sc - sc.max())
            sc /= sc.sum()
            out[hi] = sc @ Vh
        x = x + W[f"L{li}.Wo"] @ out.reshape(-1)
        h = rmsnorm(x, W[f"L{li}.ffn_norm"], RMS_EPS)
        g = W[f"L{li}.ffn_gate"] @ h
        u = W[f"L{li}.ffn_up"]   @ h
        x = x + W[f"L{li}.ffn_down"] @ (silu(g) * u)
    x = rmsnorm(x, W["output_norm"], RMS_EPS)
    last = W["output"] @ x

with open(OUT_REF, 'wb') as f:
    f.write(struct.pack('<i', len(prompt)))
    f.write(prompt.tobytes())
    f.write(struct.pack('<i', VOCAB))
    f.write(last.astype(np.float32).tobytes())
print(f"wrote {OUT_REF}: prompt_len={len(prompt)} vocab={VOCAB}")
print(f"last_logits[:5] = {last[:5]}")
