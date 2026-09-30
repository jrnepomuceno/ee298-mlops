# Area 2 — Optimized Training → ONNX (Windows · RTX 3060 Ti 8 GB)

Goal: train the VCM on the 3060 Ti as fast as 8 GB VRAM allows, then export
and measure an int8 ONNX for the Pi5. This is the full, runnable procedure — the
5-line outline it replaced is obsolete.

Everything below maps to real flags in `model/main.py` and the export/quantize
scripts. Defaults shown in parentheses.

---

## 0. What already exists (do not rebuild)

| Capability | Where | Status |
|------------|-------|--------|
| bf16 AMP (cuda only, no GradScaler) | `model/train.py::train_one_epoch` | ✅ `--amp` flag |
| Precomputed-mel mmap (no per-epoch audio decode) | `model/dataset.py` + `model/precompute_mels.py` | ✅ `--mels-dir` |
| Mel-domain noise augmentation (train only) | `model/dataset.py` | ✅ `--noise-snr` / `--noise-dir` |
| Gradient clipping (5.0) | `model/train.py` | ✅ on by default |
| 4 known-bug fixes (OOV CTC mask, true input lengths, …) | `model/train.py::_ctc_loss` | ✅ applied |
| FP32 ONNX export + ORT graph optimize | `model/export_onnx.py`, `model/optimize_onnx.py` | ✅ |
| **int8 quantization** | `model/quantize_onnx.py` | ✅ **added this session** |

The int8 quantization step is implemented in `model/quantize_onnx.py`; artifact
size is reported but has no fixed cap for now.

---

## 1. Environment (Windows)

```bat
:: from the me2/ project root
py -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install onnx onnxruntime numpy soundfile
nvidia-smi        :: confirm the 3060 Ti + driver
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True NVIDIA GeForce RTX 3060 Ti`. If `False`, the CUDA wheel/driver
pair is wrong — fix before training or `--amp` silently no-ops on CPU.

---

## 2. Data

Two paths. **Use the real-data path for the production model**; synthetic is
only for wiring checks.

### 2a. Real data (recommended)
Manifest = JSONL (or CSV) with `path`, `intent`, `transcript` per row.
```bat
:: one-time: precompute log-mels so training never decodes audio again
python model\precompute_mels.py --manifest data\real_manifest.jsonl --out-dir data\mels --workers 8
```
Writes `mels_{train,val,test}.npy` + `_lengths.npy` + sidecars. Cached by
manifest fingerprint — reruns skip unchanged splits.

### 2b. Synthetic (smoke test only)
```bat
python model\main.py generate --num-samples 2000
```

---

## 3. Train (the 8 GB-optimized recipe)

```bat
python model\main.py train ^
  --manifest data\real_manifest.jsonl ^
  --mels-dir data\mels ^
  --model-size baseline ^
  --batch-size 64 ^
  --amp ^
  --num-workers 4 ^
  --pin-memory ^
  --noise-snr 15 ^
  --noise-dir data\ambient_noise ^
  --lr 3e-4 --weight-decay 1e-4 --dropout 0.3 ^
  --epochs 30 ^
  --device cuda ^
  --output-dir checkpoints ^
  --log-file checkpoints\train.log
```

### Why these values (8 GB VRAM math)
- **`--batch-size 64`** (default 32 → doubled). Activations dominate VRAM, not
  weights (1.86 M params ≈ 7.4 MB fp32). At T≤400, B=64 with bf16 autocast
  lands ~4–5 GB; headroom to **B=96** if `nvidia-smi` shows < 6 GB used.
  Raise in steps of 32 until you see OOM, then back off one.
- **`--amp`** — bf16 autocast halves activation memory and speeds the conv/GRU
  on the Turing-tensor-core 3060 Ti. Backward/optimizer stay fp32 (no scaler
  needed for bf16).
- **`--mels-dir`** — the single biggest throughput win. Without it every epoch
  re-decodes WAV + recomputes mels on the CPU and starves the GPU. With it,
  batches are read straight from `.npy`.
- **`--num-workers 4 --pin-memory`** — 4 worker processes feed the GPU; pinned
  memory makes the H2D copy async so the GPU never waits on pageable RAM.
  (Keep `0` on the Pi; 4 is for the dev box.)
- **`--noise-snr 15 [--noise-dir …]`** — train-only mel-domain noise
  ([15, 25] dB). Cheap robustness gain for real-room recordings. Omit
  `--noise-dir` for white Gaussian, or point it at speech-kws
  `_background_noise_` for realistic ambience.
- **`--epochs 30`** — the model is small; give it room to converge. Watch
  `val_macro_f1` / `val_wer` in the log and stop early if flat for ~5 epochs
  (no built-in patience flag — read `checkpoints\history.json`).

### Watch
`nvidia-smi` (VRAM + util should be high, not pegged at 0 % = data-starved)
and `checkpoints\train.log`. Best checkpoint → `checkpoints\pi5-vcm-best.pt`
(saved on val-loss improvement).

---

## 4. Evaluate

```bat
python model\main.py test --ckpt checkpoints\pi5-vcm-best.pt
```
Read **`test_macro_f1`** (intent) and **`test_wer`** (slot/CTC). These are the
numbers that decide whether the Area 1 architecture (kept as-is) is genuinely
sufficient on real data — the open question from Area 1.

Acceptance bar (proposal, tune to product needs):
- intent macro-F1 ≥ 0.90 overall, ≥ 0.85 per non-OOV intent
- slot WER ≤ 0.15 on the constrained vocab
- OOV recall high (the safety class — mis-firing a command is worse than a miss)

---

## 5. Export → optimize → int8

```bat
:: 1) FP32 ONNX from the best checkpoint
python model\export_onnx.py --checkpoint checkpoints\pi5-vcm-best.pt --output vcm_model_fp32.onnx

:: 2) ORT graph optimization (fuse/prune)
python model\optimize_onnx.py --input vcm_model_fp32.onnx --output optimized_model.onnx

:: 3) int8 dynamic quantization  <-- the step that was missing
python model\quantize_onnx.py --input optimized_model.onnx --output vcm_model_int8.onnx --verify
```

Expected sizes:
| Artifact | Size | Budget |
|----------|------|--------|
| `vcm_model_fp32.onnx` | ~7.6 MB | (intermediate) |
| `vcm_model_int8.onnx` | model-dependent; measure after export | informational |

`--verify` runs a desktop-CPU latency proxy (informational only — **re-time on
the Pi5**, the Cortex-A76 is the real target; that's Area 3).

Ship `vcm_model_int8.onnx` to the Pi. The Pi inference path (Area 3) loads this
via ONNX Runtime CPU EP.

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `torch.cuda.is_available() == False` | wrong CUDA wheel / driver | reinstall cu121 wheels; update driver |
| OOM at B=64 | activations too big | drop to 48/32, or `--max-frames` lower if data allows |
| GPU util ~0 %, slow | data-starved | ensure `--mels-dir` set; raise `--num-workers`; `--pin-memory` |
| `--amp` no speedup | running on CPU | confirm `--device cuda` + CUDA available |
| val_wer stuck high | CTC vocab / transcripts | check `transcript_to_tokens` coverage; OOV masking is on |
| int8 file unexpectedly large | quantization may not have applied as expected | inspect ONNX operators and compare with the FP32 file; there is no fixed size gate |

---

## 7. Deliverables this area produces
- `checkpoints\pi5-vcm-best.pt` — best PyTorch checkpoint
- `checkpoints\history.json` — per-epoch metrics
- `vcm_model_int8.onnx` — the Pi5 deployment artifact; record its measured size
- `docs\training-report.md` — copy the final TRAINING REPORT block here

Next: **Area 3** — point the Pi inference path at `vcm_model_int8.onnx` and
benchmark RTF ≤ 0.3 on the actual device.
