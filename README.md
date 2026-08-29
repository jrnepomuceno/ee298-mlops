# EE298 — Machine Learning Operations (MLOps)

**Machine exercises** · Joven Daniel Nepomuceno

This repository hosts the machine exercises for **EE298 (MLOps)**. Each
exercise is a self-contained folder named `me<N>/` with its own README,
source code, executed notebooks, and model artifacts, so any exercise can be
cloned, reproduced, and reviewed independently of the others.

---

## Repository layout

```
ee298-mlops/
├── README.md            # this file — course-level overview
├── .gitignore           # venvs, datasets, caches, OS cruft
└── me1/                 # Machine Exercise 1 (complete)
    ├── README.md        # full documentation for the exercise
    ├── model.py         # einops/einsum CNN
    ├── train.py         # DDP/AMP training script
    ├── mnist_einsum_cnn.ipynb      # executed notebook
    └── mnist_cnn_einsum_best.pt    # best checkpoint
```

---

## Exercise index

| # | Folder | Title | Status | Headline result |
|---|--------|-------|--------|-----------------|
| 1 | [`me1/`](me1/) | MNIST CNN written entirely with `einops` / `einsum` | ✅ Complete | 97.22% test accuracy (5 epochs) |
| 2 | `me2/` | _TBD_ | ⏳ Planned | — |
| 3 | `me3/` | _TBD_ | ⏳ Planned | — |

New exercises are added as sibling folders (`me2/`, `me3/`, …) and registered
in the table above.

---

## Machine Exercise 1 — at a glance

**Task.** Build a 3-layer CNN for MNIST in which **every layer is expressed
explicitly** with `einops` (`rearrange` / `reduce`) and `torch.einsum` — no
`F.conv2d`, no `F.max_pool2d`, no `F.linear`, no `F.pad` — then train it and
verify the hand-written layers numerically against PyTorch's reference ops.

**Artifacts**

| File | Role |
|---|---|
| `me1/model.py` | einops/einsum architecture + numerical self-check vs. reference ops |
| `me1/train.py` | Cluster-ready training loop (single device → 8×A100 → multi-node Slurm; DDP + AMP) |
| `me1/mnist_einsum_cnn.ipynb` | Executed notebook: model, training, results, 4×4 prediction grid |
| `me1/mnist_cnn_einsum_best.pt` | Best checkpoint (epoch 4, 97.22% test accuracy) |

**Quick start**

```bash
cd me1
python -m venv my_venv && source my_venv/bin/activate
pip install torch torchvision einops
python train.py --epochs 5    # ~86 s/epoch on Apple Silicon (MPS)
```

Full details — architecture walkthrough, layer-by-layer einsum mapping, CLI
reference, per-epoch results, and reproducibility notes — live in
[`me1/README.md`](me1/README.md).

---

## Conventions

These rules apply to every exercise folder:

- **Self-contained exercises.** Each `me<N>/` ships its own README, code, and
  artifacts; nothing is imported across exercise folders.
- **Datasets are never committed.** Data (e.g. MNIST, ~65 MB) is downloaded
  by torchvision on first run and cached under `me<N>/data/` (git-ignored).
- **Virtualenvs are never committed.** Each exercise creates `my_venv/`
  locally (git-ignored); dependencies are small and listed in the exercise
  README.
- **Checkpoints are committed.** Best-validation checkpoints are saved as
  `<task>_best.pt` and committed alongside the code, so results can be
  inspected without retraining.
- **Notebooks are committed executed.** Notebooks ship with outputs embedded
  (plots, tables, prediction grids) so they are reviewable without running.
- **Device portability.** Training scripts auto-select CUDA → MPS (Apple
  Silicon) → CPU, so the same code runs on a laptop, a workstation, or a
  cluster unchanged.

---

## Requirements

| Package | Minimum version | Used by |
|---|---|---|
| Python | 3.10 | all exercises |
| `torch` | 2.0 | all (DDP/AMP in `train.py`) |
| `torchvision` | any recent | dataset loading |
| `einops` | any recent | `me1` layer definitions |
| `matplotlib` | any | notebook plots |

---

## Changelog

| Date | Change |
|---|---|
| 2026-08-29 | `me1` completed: model, training, executed notebook, best checkpoint, exercise README |
| 2026-08-29 | Root README rewritten as course-level overview; repository initialized |
