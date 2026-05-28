#!/usr/bin/env python3
"""Generate a numpy reference LLaMA-arch forward pass on the SAME weights
that tools/make_tiny_llama_gguf.py emits, for a fixed prompt.  The C test
reads /tmp/slate_llama_ref.bin and compares its slate_llama_prefill
output to this reference within fp32 tolerance.

This is what proves slate's LLaMA inference matches the spec —
without it we'd be writing kernels in the dark.
"""
import struct, numpy as np

# Mirror the fixture's hyperparameters.
N_LAYERS, D, N_HEADS, FFN, VOCAB, MAX_SEQ = 2, 32, 2, 64, 64, 128
HEAD_DIM = D // N_HEADS
N_KV_HEADS = N_HEADS
RMS_EPS = 1e-5
THETA_BASE = 10000.0

rng = np.random.default_rng(0xCAFE)
def w(shape):
    return rng.normal(0, 0.02, shape).astype(np.float32)

# Build the same weight set in the same order as make_tiny_llama_gguf.py.
weights = {}
weights["token_embd"]  = w((VOCAB, D))
weights["output_norm"] = np.ones((D,), dtype=np.float32)
weights["output"]      = w((VOCAB, D))
for i in range(N_LAYERS):
    weights[f"L{i}.attn_norm"]   = np.ones((D,), dtype=np.float32)
    # llama.cpp convention: linear weights are stored [out_dim, in_dim].
    weights[f"L{i}.Wq"]          = w((D, D))
    weights[f"L{i}.Wk"]          = w((N_KV_HEADS * HEAD_DIM, D))
    weights[f"L{i}.Wv"]          = w((N_KV_HEADS * HEAD_DIM, D))
    weights[f"L{i}.Wo"]          = w((D, D))
    weights[f"L{i}.ffn_norm"]    = np.ones((D,), dtype=np.float32)
    weights[f"L{i}.ffn_gate"]    = w((FFN, D))
    weights[f"L{i}.ffn_up"]      = w((FFN, D))
    weights[f"L{i}.ffn_down"]    = w((D,   FFN))

def rmsnorm(x, w, eps):
    rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return (x / rms) * w

def silu(x):
    return x / (1.0 + np.exp(-x))

def rope(x, position, theta_base, head_dim):
    # x shape: [n_heads, head_dim]
    half = head_dim // 2
    inv_freq = theta_base ** (-2.0 * np.arange(half) / head_dim)
    angle = position * inv_freq                              # [half]
    c, s = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
    x = x.copy()
    a = x[..., :half].copy()
    b = x[..., half:].copy()
    x[..., :half] = a * c - b * s
    x[..., half:] = a * s + b * c
    return x

def forward_step(x, position, K_cache, V_cache):
    # x: [D]
    for li in range(N_LAYERS):
        # attn pre-norm
        h = rmsnorm(x, weights[f"L{li}.attn_norm"], RMS_EPS)
        # Q, K, V
        q = weights[f"L{li}.Wq"] @ h       # [D]
        k = weights[f"L{li}.Wk"] @ h       # [n_kv_heads*head_dim]
        v = weights[f"L{li}.Wv"] @ h
        # Reshape to per-head and apply RoPE
        q_h = q.reshape(N_HEADS,    HEAD_DIM)
        k_h = k.reshape(N_KV_HEADS, HEAD_DIM)
        q_h = rope(q_h, position, THETA_BASE, HEAD_DIM)
        k_h = rope(k_h, position, THETA_BASE, HEAD_DIM)
        # Append to cache at slot `position`
        K_cache[li, position] = k_h.reshape(-1)
        V_cache[li, position] = v
        # Multi-head attention against cache positions 0..position
        L = position + 1
        attn_out = np.zeros((N_HEADS, HEAD_DIM), dtype=np.float32)
        for hi in range(N_HEADS):
            q_row = q_h[hi]                                              # [hd]
            K_h = K_cache[li, :L].reshape(L, N_KV_HEADS, HEAD_DIM)[:, hi] # [L, hd]
            V_h = V_cache[li, :L].reshape(L, N_KV_HEADS, HEAD_DIM)[:, hi] # [L, hd]
            scores = (K_h @ q_row) / np.sqrt(HEAD_DIM)                   # [L]
            mx = scores.max()
            p_w = np.exp(scores - mx)
            p_w = p_w / p_w.sum()
            attn_out[hi] = p_w @ V_h
        a = weights[f"L{li}.Wo"] @ attn_out.reshape(-1)                  # [D]
        x = x + a
        # FFN pre-norm
        h = rmsnorm(x, weights[f"L{li}.ffn_norm"], RMS_EPS)
        g = weights[f"L{li}.ffn_gate"] @ h
        u = weights[f"L{li}.ffn_up"]   @ h
        f = silu(g) * u
        d = weights[f"L{li}.ffn_down"] @ f
        x = x + d
    x = rmsnorm(x, weights["output_norm"], RMS_EPS)
    logits = weights["output"] @ x                                       # [VOCAB]
    return logits, K_cache, V_cache

# Fixed prompt
prompt = np.array([1, 7, 13, 25, 3], dtype=np.int32)
K_cache = np.zeros((N_LAYERS, MAX_SEQ, N_KV_HEADS * HEAD_DIM), dtype=np.float32)
V_cache = np.zeros((N_LAYERS, MAX_SEQ, N_KV_HEADS * HEAD_DIM), dtype=np.float32)
last_logits = None
for p_i, tok in enumerate(prompt):
    x = weights["token_embd"][tok].copy()
    last_logits, K_cache, V_cache = forward_step(x, p_i, K_cache, V_cache)

# Dump for the C test
with open('/tmp/slate_llama_ref.bin', 'wb') as f:
    f.write(struct.pack('<i', len(prompt)))
    f.write(prompt.tobytes())
    f.write(struct.pack('<i', VOCAB))
    f.write(last_logits.astype(np.float32).tobytes())
print(f"wrote /tmp/slate_llama_ref.bin: prompt_len={len(prompt)} vocab={VOCAB}")
print(f"last_logits[:5] = {last_logits[:5]}")
