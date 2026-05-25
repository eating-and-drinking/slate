# 03_tinyshakespeare — char-level mini-GPT

A single-head, 4-layer transformer trained on the tinyshakespeare corpus.

## Getting the corpus

```bash
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

About 1 MB of Shakespeare text, ~65 unique characters.

## Running

```bash
./build/examples/03_tinyshakespeare/slate_tinyshakespeare input.txt
```

Expected: training loss drops from ~log(65) ≈ 4.17 toward ~1.5 over 5000 steps
(several minutes on a modern laptop, no SIMD yet — M3 will accelerate this).
After training the example greedy-decodes 200 chars starting from "ROMEO:".

## Architecture

```
Config:    SEQ=128, D_MODEL=128, N_LAYERS=4, FFN_H=256, vocab~65
Params:    ~1.0M
Single-head causal attention with learned positional embedding.
Pre-norm RMSNorm, SwiGLU FFN.
```

## Caveats (M2)

- Greedy decoding only. Temperature sampling will land in M3 alongside SIMD.
- Single-head; multi-head awaits M3 too.
- No RoPE; we use a learned positional embedding. For longer-context
  models we'll switch to RoPE in M4.
- No SIMD; expect ~50-200 ms per step depending on hardware. SIMD in M3
  should bring 5-10x.
