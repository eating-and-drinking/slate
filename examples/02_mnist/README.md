# 02_mnist — MLP on MNIST

A three-layer fully connected network (784 → 128 → 64 → 10) trained with
AdamW + cosine LR schedule + gradient clipping + cross-entropy loss.

## Getting the data

The example expects the four classic IDX files. The original Yann LeCun mirror
is intermittently unavailable; any of these work:

- HuggingFace: <https://huggingface.co/datasets/ylecun/mnist/tree/main>
- The `torchvision.datasets.MNIST` cache (look under `~/data/MNIST/raw/`)
- Manually from <https://ossci-datasets.s3.amazonaws.com/mnist/>

Place them in one directory:

```
~/data/mnist/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

If your downloads are gzipped, decompress them first (`gunzip *.gz`).

## Running

```bash
./build/examples/02_mnist/slate_mnist ~/data/mnist
```

## Expected output

```
slate 0.1.0 — mnist example
[mnist] train: 60000 images, test: 10000 images
[mnist] 6 parameter tensors
[mnist] epoch  0  loss=0.4123  test_acc=0.9520
[mnist] epoch  1  loss=0.2014  test_acc=0.9651
...
[mnist] epoch  9  loss=0.0334  test_acc=0.9712
[mnist] done in 312.4s
```

Acceptance: test accuracy ≥ 97% after 10 epochs (M1 acceptance criterion).

## Performance notes

This example does not yet use SIMD or threading; that arrives in M3. On a 2020
laptop expect 30–60 seconds per epoch in `-O2`. With `-O3 -march=native` you
get roughly 2× speedup. SIMD/threading is expected to deliver another 5–10×
in M3.
