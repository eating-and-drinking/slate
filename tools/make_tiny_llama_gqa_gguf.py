#!/usr/bin/env python3
"""GQA LLaMA-format GGUF: n_q_heads = 8, n_kv_heads = 2 (4× reduction).

Real-world: LLaMA-3-8B has n_q_heads=32, n_kv_heads=8 (4× reduction);
LLaMA-3-70B has n_q_heads=64, n_kv_heads=8 (8× reduction).
This fixture mirrors that pattern at smaller scale.
"""
import struct, numpy as np

OUT_GGUF = '/tmp/slate_tiny_llama_gqa.gguf'
OUT_REF  = '/tmp/slate_tiny_llama_gqa_ref.bin'

N_LAYERS, D, N_HEADS, FFN, VOCAB, MAX_SEQ = 2, 64, 8, 128, 64, 128
N_KV_HEADS = 2                # GQA: 8 Q heads share 2 K/V heads (group=4)
HEAD_DIM = D // N_HEADS
RMS_EPS = 1e-5
THETA_BASE = 10000.0

T_U32, T_STRING, T_F32 = 4, 8, 6
GGML_T_F32 = 0
def w_u32(b, v): b.extend(struct.pack('<I', v))
def w_u64(b, v): b.extend(struct.pack('<Q', v))
def w_f32(b, v): b.extend(struct.pack('<f', v))
def w_str(b, s):
    s = s.encode(); w_u64(b, len(s)); b.extend(s)
def kv_u32(b,k,v): w_str(b,k); w_u32(b,T_U32); w_u32(b,v)
def kv_f32(b,k,v): w_str(b,k); w_u32(b,T_F32); w_f32(b,v)
def kv_str(b,k,v): w_str(b,k); w_u32(b,T_STRING); w_str(b,v)

rng = np.random.default_rng(0xCAFE)
def w(shape): return rng.normal(0, 0.02, shape).astype(np.float32)

# llama.cpp convention: linear weights stored [out_dim, in_dim].
W = {}
W["token_embd"]  = w((VOCAB, D))
W["output_norm"] = np.ones((D,), dtype=np.float32)
W["output"]      = w((VOCAB, D))
KV_DIM = N_KV_HEADS * HEAD_DIM
for i in range(N_LAYERS):
    W[f"L{i}.attn_norm"] = np.ones((D,), dtype=np.float32)
    W[f"L{i}.Wq"]        = w((D,      D))
    W[f"L{i}.Wk"]        = w((KV_DIM, D))   # GQA: smaller K
    W[f"L{i}.Wv"]        = w((KV_DIM, D))   # GQA: smaller V
    W[f"L{i}.Wo"]        = w((D,      D))
    W[f"L{i}.ffn_norm"]  = np.ones((D,), dtype=np.float32)
    W[f"L{i}.ffn_gate"]  = w((FFN, D))
    W[f"L{i}.ffn_up"]    = w((FFN, D))
    W[f"L{i}.ffn_down"]  = w((D,   FFN))

# Map slate name → GGUF name
gguf_name = {
    "token_embd":  "token_embd.weight",
    "output_norm": "output_norm.weight",
    "output":      "output.weight",
}
for i in range(N_LAYERS):
    pfx_g = f"blk.{i}."
    for s, g in [("attn_norm","attn_norm"), ("Wq","attn_q"), ("Wk","attn_k"),
                  ("Wv","attn_v"), ("Wo","attn_output"), ("ffn_norm","ffn_norm"),
                  ("ffn_gate","ffn_gate"), ("ffn_up","ffn_up"), ("ffn_down","ffn_down")]:
        gguf_name[f"L{i}.{s}"] = pfx_g + g + ".weight"

# Build GGUF
buf = bytearray(b'GGUF')
w_u32(buf, 3)
w_u64(buf, len(W))
w_u64(buf, 11)
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

# Tensor info
ti_payload = bytearray(); offsets = []
def pad(n, a):
    r = n % a; return n if r == 0 else n + (a - r)
cur = 0
ordered = list(W.items())
for name, arr in ordered:
    payload = arr.tobytes()
    cur = pad(cur, 32); offsets.append(cur)
    ti_payload.extend(b'\0' * (cur - len(ti_payload))); ti_payload.extend(payload)
    cur = len(ti_payload)
for (name, arr), off in zip(ordered, offsets):
    w_str(buf, gguf_name[name])
    w_u32(buf, arr.ndim)
    for d in arr.shape[::-1]: w_u64(buf, d)
    w_u32(buf, GGML_T_F32)
    w_u64(buf, off)
while len(buf) % 32 != 0: buf.append(0)
buf += ti_payload
with open(OUT_GGUF, 'wb') as f: f.write(buf)
print(f"wrote {OUT_GGUF}: {len(buf)} bytes (GQA: {N_HEADS} Q heads, {N_KV_HEADS} KV heads, group={N_HEADS//N_KV_HEADS})")

# Numpy reference forward
def rmsnorm(x, ws, eps):
    return (x / np.sqrt((x*x).mean(axis=-1, keepdims=True) + eps)) * ws
def silu(x): return x / (1.0 + np.exp(-x))
def rope(x, pos, theta, hd):
    half = hd // 2
    inv = theta ** (-2.0 * np.arange(half) / hd)
    a = pos * inv
    c, s = np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)
    x = x.copy()
    lo, hi = x[..., :half].copy(), x[..., half:].copy()
    x[..., :half] = lo*c - hi*s
    x[..., half:] = lo*s + hi*c
    return x

group = N_HEADS // N_KV_HEADS
prompt = np.array([1, 7, 13, 25, 3], dtype=np.int32)
K_c = np.zeros((N_LAYERS, MAX_SEQ, KV_DIM), dtype=np.float32)
V_c = np.zeros_like(K_c)
last = None
for pi, tok in enumerate(prompt):
    x = W["token_embd"][tok].copy()
    for li in range(N_LAYERS):
        h = rmsnorm(x, W[f"L{li}.attn_norm"], RMS_EPS)
        q = W[f"L{li}.Wq"] @ h
        k = W[f"L{li}.Wk"] @ h
        v = W[f"L{li}.Wv"] @ h
        q_h = q.reshape(N_HEADS, HEAD_DIM)
        k_h = k.reshape(N_KV_HEADS, HEAD_DIM)
        q_h = rope(q_h, pi, THETA_BASE, HEAD_DIM)
        k_h = rope(k_h, pi, THETA_BASE, HEAD_DIM)
        K_c[li, pi] = k_h.reshape(-1); V_c[li, pi] = v
        L = pi + 1
        out = np.zeros((N_HEADS, HEAD_DIM), dtype=np.float32)
        for hq in range(N_HEADS):
            hkv = hq // group
            Kh = K_c[li, :L].reshape(L, N_KV_HEADS, HEAD_DIM)[:, hkv]
            Vh = V_c[li, :L].reshape(L, N_KV_HEADS, HEAD_DIM)[:, hkv]
            sc = (Kh @ q_h[hq]) / np.sqrt(HEAD_DIM)
            sc = np.exp(sc - sc.max()); sc /= sc.sum()
            out[hq] = sc @ Vh
        x = x + W[f"L{li}.Wo"] @ out.reshape(-1)
        h = rmsnorm(x, W[f"L{li}.ffn_norm"], RMS_EPS)
        g = W[f"L{li}.ffn_gate"] @ h
        u = W[f"L{li}.ffn_up"]   @ h
        x = x + W[f"L{li}.ffn_down"] @ (silu(g) * u)
    last = W["output"] @ rmsnorm(x, W["output_norm"], RMS_EPS)

with open(OUT_REF, 'wb') as f:
    f.write(struct.pack('<i', len(prompt)))
    f.write(prompt.tobytes())
    f.write(struct.pack('<i', VOCAB))
    f.write(last.astype(np.float32).tobytes())
print(f"wrote {OUT_REF}: prompt_len={len(prompt)} vocab={VOCAB}")
print(f"last_logits[:5] = {last[:5]}")
