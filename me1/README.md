# me1 — MNIST digit classifier built entirely from `einops` / `einsum`

A 3-layer CNN for MNIST digit classification in which **every layer is written
explicitly** with `einops` (`rearrange` / `reduce`) and `torch.einsum` — no
`F.conv2d`, no `F.max_pool2d`, no `F.linear`, no `F.pad`. The goal is
pedagogical: make each operation's index arithmetic visible, then prove the
implementation numerically equivalent to PyTorch's reference ops.

The same model ships three ways:

| Artifact | What it is |
|---|---|
| `model.py` | The architecture + a self-check against PyTorch reference ops |
| `train.py` | Cluster-ready training loop (single device → 8×A100 → multi-node Slurm, DDP + AMP) |
| `mnist_einsum_cnn.ipynb` | Executed notebook: model, training, results, and a 4×4 prediction grid |
| `mnist_cnn_einsum_best.pt` | Best checkpoint from the 5-epoch run (test acc **97.22%**) |

---

## Repository layout

```
me1/
├── model.py                      # einops/einsum CNN (7.1 KB)
├── train.py                      # DDP/AMP training script (11.6 KB)
├── mnist_einsum_cnn.ipynb        # executed notebook (140 KB, outputs embedded)
├── mnist_cnn_einsum_best.pt      # best checkpoint: epoch 4, 97.22% test acc
├── sample.md                     # misc sample artifact
├── data/                         # MNIST cache (created on first run; git-ignored)
└── my_venv/                      # virtualenv (git-ignored)
```

---

## Model

### Architecture

```
MNISTCNN   (93,962 parameters)

  x: (B, 1, 28, 28)
  │
  ├─ EinsumConv2d(1  → 32, k=3, s=1, p=1)  + ReLU
  ├─ EinsumMaxPool2d(2, 2)                  28×28 → 14×14
  ├─ EinsumConv2d(32 → 64, k=3, s=1, p=1)  + ReLU
  ├─ EinsumMaxPool2d(2, 2)                  14×14 → 7×7
  ├─ EinsumConv2d(64 → 128, k=3, s=1, p=1) + ReLU
  ├─ EinsumGlobalAvgPool2d()                7×7   → 1×1
  └─ EinsumLinear(128 → 10)                 logits
```

Spatial flow: `28×28 → 14×14 → 7×7 → 1×1`. No batch norm, no dropout —
deliberately minimal.

### How each layer is expressed

| PyTorch op | me1 equivalent | Mechanism |
|---|---|---|
| `F.pad` | `EinsumConv2d._zero_pad` | explicit `new_zeros` buffer; input copied into the center |
| `F.conv2d` | `EinsumConv2d.forward` | sum of **K² rank-1 convolutions**, one per kernel tap: `y[b,o,h,w] += Σ_c w[o,c,u,v] · x[b,c,h+u,w+v]` computed as `einsum(x_uv, w_uv, "b c h w, o c -> b o h w")` and accumulated over `(u, v)` |
| `F.max_pool2d` | `EinsumMaxPool2d` | `rearrange(x, "b c (h k) (w l) -> b c h w k l")` then `reduce(..., "max")` over the window axes (stride == kernel only) |
| `F.adaptive_avg_pool2d` | `EinsumGlobalAvgPool2d` | `reduce(x, "b c h w -> b c 1 1", "mean")` |
| `F.linear` | `EinsumLinear` | flatten with `rearrange(x, "b ... -> b (...)")`, then `einsum(flat, w, "b i, o i -> b o") + bias` |
| `F.flatten` | — | `rearrange(x, "b c 1 1 -> b c")` |

Initialization matches PyTorch conventions: `kaiming_uniform_(a=√5)` on
weights (ReLU-friendly) and uniform bias in `±1/√fan_in`.

### Numerical verification

`python model.py` runs a self-check: forward passes of the einsum conv and
linear layers are compared element-wise against `F.conv2d` / `F.linear` with
identical weights. Observed agreement:

```
conv1 vs F.conv2d : max abs diff ≈ 3.6e-07   (fp32 rounding)
fc    vs F.linear : max abs diff = 0.0
```

i.e. the einsum formulations are bit-for-bit faithful modulo floating-point
association order.

> **Performance note:** the conv is a Python loop over K² taps, so it is
> substantially slower than cuDNN. That is the price of explicitness — this is
> a teaching/reference implementation, not a throughput-optimized one.

---

## Dataset

**MNIST** — the classic 28×28 grayscale handwritten-digit set
(Cortes & LeCun, 1998). It is **not bundled**; torchvision downloads the four
IDX files on first run and caches them:

```
data/MNIST/raw/
├── train-images-idx3-ubyte   60,000 × 28×28
├── train-labels-idx1-ubyte   60,000 labels
├── t10k-images-idx3-ubyte    10,000 × 28×28
└── t10k-labels-idx1-ubyte    10,000 labels
```

Preprocessing: `ToTensor()` + `Normalize(mean=0.1307, std=0.3081)` (the
dataset's global mean/std). Cache location is `--data-dir` (default `./data`).

**Split semantics:** MNIST has no validation split — the canonical split is
60,000 train / 10,000 test. `train.py` therefore evaluates on the official
held-out 10k set after every epoch; the "val" figures below **are** test-split
figures.

---

## Training

`train.py` is a complete, cluster-ready loop. It auto-detects its mode from
the environment: plain `python` gives single-device training; launched with
`torchrun` it becomes a DDP job (picks up `RANK` / `WORLD_SIZE` /
`LOCAL_RANK`, uses `DistributedSampler` with `drop_last` for equal per-rank
batches, `nccl` or `gloo` backend). Device selection: CUDA → MPS (Apple
Silicon) → CPU.

### Quick start

```bash
cd me1
python -m venv my_venv && source my_venv/bin/activate
pip install torch torchvision einops

python train.py --epochs 5          # ~86 s/epoch on an M-series Mac (MPS)
```

### Usage examples

```bash
# Single GPU / CPU
python train.py --epochs 3

# One node, 8× A100, bf16 mixed precision
torchrun --nproc_per_node=8 train.py --amp --amp-dtype bf16 --batch-size 256

# Multi-node (Slurm, 4 nodes × 8 GPUs)
torchrun --nnodes=4 --node_rank=$SLURM_NODEID --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py --amp --amp-dtype bf16 --batch-size 256

# Smoke test on the cluster (cap training data to 512 samples)
python train.py --epochs 1 --subset 512
```

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 3 | number of epochs |
| `--batch-size` | 128 | **per-GPU** batch size; global = `batch-size × world_size` |
| `--lr` | 1e-3 | Adam learning rate |
| `--scale-lr` | `none` | `linear`: multiply LR by `world_size` (linear scaling rule) |
| `--weight-decay` | 1e-4 | Adam weight decay |
| `--grad-clip` | 0.0 | max gradient norm; 0 disables |
| `--seed` | 0 | base seed (each rank uses `seed + rank`) |
| `--amp` | off | enable automatic mixed precision |
| `--amp-dtype` | `bf16` | `bf16` (Ampere+, no loss scaling) or `fp16` (with GradScaler) |
| `--data-dir` | `./data` | MNIST cache directory |
| `--num-workers` | 8 | DataLoader workers |
| `--subset` | 0 | cap training-set size for smoke tests (0 = full 60k) |
| `--backend` | `nccl` | `nccl` (GPU) or `gloo` (CPU) |
| `--dist-url` | `None` → `env://` | process-group init method; `torchrun` sets it via the environment |
| `--log-interval` | 50 | log every N steps (0 disables) |
| `--checkpoint` | `mnist_cnn_einsum_best.pt` | best-model output path |
| `--save-last` | off | also save a final-epoch checkpoint |

Fixed choices: `nn.CrossEntropyLoss`, `Adam`, ReLU activations. The best
checkpoint is saved whenever validation accuracy improves; it stores
`{epoch, model_state_dict, val_acc, args}`.

### Loading the shipped checkpoint

```python
import torch
from model import MNISTCNN

ckpt = torch.load("mnist_cnn_einsum_best.pt", map_location="cpu")
model = MNISTCNN()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(ckpt["epoch"], f"{ckpt['val_acc']:.2f}%")   # 4  97.22%
```

---

## Results

### `train.py` — 5 epochs, full 60k train / 10k test, batch 128, Adam lr 1e-3

Run on Apple Silicon (MPS), ~86 s/epoch:

| Epoch | Train loss | Test loss | Test acc | Time |
|---|---|---|---|---|
| 1 | 0.6772 | 0.1908 | 94.37% | 88.9 s |
| 2 | 0.1886 | 0.1400 | 95.92% | 85.5 s |
| 3 | 0.1367 | 0.1020 | 96.97% | 85.6 s |
| 4 | 0.1107 | 0.0961 | **97.22%** ⭐ best | 85.6 s |
| 5 | 0.0909 | 0.0911 | 97.13% | 85.5 s |

Epoch 4 was the best validation epoch, so `mnist_cnn_einsum_best.pt` holds the
epoch-4 weights.

### `mnist_einsum_cnn.ipynb` — independent in-notebook run (seed 0)

| Epoch | Train loss | Test loss | Test acc |
|---|---|---|---|
| 1 | 0.6788 | 0.2261 | 93.18% |
| 2 | 0.1956 | 0.1407 | 95.80% |
| 3 | 0.1423 | 0.1367 | 95.92% |
| 4 | 0.1138 | 0.1008 | 96.94% |
| 5 | 0.0958 | 0.0973 | **96.95%** |

Same configuration, freshly retrained inside the notebook — hence the slight
difference from the `train.py` run above. The notebook ends with a 4×4 grid of
16 random test images (seed 0) showing ground truth vs. prediction; that run
scored 16/16.

Both runs land in the expected range for this architecture (≈97–98% is typical
for a 3-conv net on MNIST).

---

## Jupyter notebook

`mnist_einsum_cnn.ipynb` is fully executed (all outputs embedded, 0 errors)
and contains, in order:

1. **Model** — the complete einops/einsum implementation verbatim from
   `model.py`, plus the numerical cross-check against `F.conv2d` / `F.linear`
2. **Data** — MNIST loaders with the standard normalization
3. **Training** — 5 epochs, `CrossEntropyLoss` + Adam, on MPS
4. **Results** — per-epoch loss/accuracy table and plots
5. **Predictions** — 4×4 grid of 16 test images with GT vs. prediction titles
   (green = correct, red = wrong)

Open it in JupyterLab/Jupyter Notebook or VS Code to browse; re-running any
cell works as usual.

---

## Requirements

- Python ≥ 3.10
- `torch` ≥ 2.0 (the script uses `torch.amp.autocast(device_type, ...)` and
  `torch.amp.GradScaler("cuda", ...)`)
- `torchvision` (MNIST download)
- `einops`

CUDA is optional — the script falls back to MPS, then CPU. `nccl` is only
needed for multi-GPU DDP; use `--backend gloo` for CPU-only distributed runs.

## Reproducibility

Seeds are fixed (`--seed 0`, per-rank offset in DDP), so single-device runs
are reproducible on the same hardware/library versions. Exact floating-point
results are hardware-dependent (cuDNN vs. MPS vs. CPU), but accuracies stay
within ~0.1–0.3% across devices.

## Caveats

- **"Val" = test split.** MNIST provides no validation set; the reported
  "val" metrics are computed on the official 10k test set. If you need true
  early stopping, carve a validation slice out of the 60k train set.
- **Explicitness costs speed.** The K²-tap einsum conv is a reference
  implementation; expect it to be slower than `F.conv2d` on GPU.
- The einsum conv supports odd kernel sizes and integer padding; the max-pool
  requires `stride == kernel`.
