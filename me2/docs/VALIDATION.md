# ME2 / VCM — On-Device Validation (Area 3)

**Date:** 2026-09-27
**Model under test:** `vcm_model_int8.onnx` (dynamic int8, weights→QInt8, activations fp32)
**Target device:** Raspberry Pi 5 (Cortex-A76 ×4, aarch64, 8 GB, 5 V @ 3 A, BT-only peripherals)
**Method:** `inference/benchmark.py` — ONNX Runtime CPU, true utterance length (no pad-to-400),
timing scoped to `session.run()` only, 10 warmup + N timed runs, ARM clock/temp sampled before/after.

These are historical measurements for the 3.46 MB artifact tested at the time. The 6 MB
limit has since been removed; size remains reported but is not a current acceptance gate.
Do not attribute these Pi measurements to the newer `v19-20260929-2` artifact without rerunning UAT.

## 1. Headline verdict

| Gate (docs/plan.md) | Limit | Measured (Pi 5) | Status |
|---|---|---|---|
| Model size (int8) | Report only; no cap | **3.46 MB** | Informational |
| RTF (inference / audio) | ≤ 0.3 | **0.014** (worst, 2 s) | ✅ PASS (~21× margin) |
| Latency p95 (1–2 s command) | < 100 ms | **28.5 ms** (2 s, 1 thread) | ✅ PASS (~3.5× margin) |
| Intent accuracy | ≥ 90% | **97.9% on-distribution** (val) | ✅ PASS (see §5 caveat) |
| Slot exact-match | — | **17.3%** (val) | ⚠️ WEAK — see §5 |

**Bottom line:** the tested int8 ONNX model is **faithful to the trained checkpoint** and is
**comfortably deployable on the Pi 5 for latency** — it clears the RTF and latency bars
by an order of magnitude and holds clock under a 60 s thermal soak. File size is recorded only.
The open risk is **quality on real, recorded speech**, not compute.

## 2. Model provenance & parity

- Re-exported a **clean, fully-dynamic** FP32 graph `vcm_model.onnx` (7.84 MB) from
  `inference/best.pt` (legacy TorchScript tracer, opset 13). The previously committed
  `optimized_model.onnx` had a shape-inference conflict (dim0 256 vs 128) that broke
  `quantize_dynamic`; the clean re-export avoids it.
- `vcm_model_int8.onnx` = dynamic int8 quantization of the clean graph → **3.46 MB**.
- **Parity vs the PyTorch checkpoint** (random-mel inputs, T ∈ {50,100,160,250}):
  intent-logit max-diff ≤ 0.086, CTC-logit max-diff ≤ 0.241, **greedy CTC decode identical
  at every length**. int8 weight-quantization adds negligible noise; the deployed model
  reproduces the checkpoint.

## 3. Latency / RTF (the number that counts = Pi 5)

Synthetic random-mel sweep, 50 runs/duration, 10 warmup.

| dur (s) | frames | mean (ms) | p50 (ms) | p95 (ms) | RTF |
|---|---|---|---|---|---|
| 0.5 | 50  | 6.18  | 6.18  | 6.22  | 0.0124 |
| 1.0 | 100 | 13.59 | 13.58 | 13.64 | 0.0136 |
| 1.5 | 150 | 21.49 | 21.49 | 21.53 | 0.0143 |
| 2.0 | 200 | 28.35 | 28.34 | 28.48 | 0.0142 |

- **Threads = 2** (safe on the Pi 5, 4 cores): RTF 0.0116, p95 23.4 ms @ 2 s — modestly
  faster than 1 thread. Recommend `intra_op_num_threads = 2` for deployment.
- **Mac (M-series, Tier A sanity only):** RTF ≈ 0.0047, p95 10.1 ms @ 2 s. Not the metric.
- **Peak RSS ≈ 93 MB** (well within the 8 GB profile).

## 4. Thermal / power (5 V @ 3 A constraint)

60 s continuous-load soak on the Pi 5:

| Metric | Before | After 60 s |
|---|---|---|
| ARM clock | 2.4 GHz | 2.4 GHz (no down-clock) |
| SoC temp | ~53 °C | 56 °C |
| `vcgencmd get_throttled` | 0x0 | **0x0 (no throttle)** |
| Post-soak 1 s command | — | 13.6 ms (unchanged) |

**Conclusion:** the model is a tiny fraction of the Pi 5's thermal/power envelope. A
5 V @ 3 A supply with Bluetooth-only peripherals is **not a bottleneck** for inference.
Even a continuous command stream would not sustain enough load to throttle.

## 5. Quality — the honest caveat

The benchmark's quality half ran on a **48-clip synthetic set** (3 speakers × 16 intents,
incl. OOV) generated with the project's toy formant TTS. Result: intent_acc 0.146,
slot F1 0.0, OOV-reject 1.0. **This is an off-distribution artifact, not a model failure:**
the test clips used different speaker IDs / seeds than the training distribution, and the
toy TTS is "formant tones, not real speech."

The **checkpoint's own validation metrics** (recorded at training, epoch 10) are the
ground truth for on-distribution quality:

| Metric | Value |
|---|---|
| val accuracy (intent) | **0.9787** |
| val macro-F1 | **0.9641** |
| **val exact-match (slots)** | **0.1729** |
| val WER / CER | 0.387 / 0.447 |
| `stop_timer` support | **0** (never in validation) |

Interpretation:
- **Intent head is strong** on the training distribution (97.9% acc, 96.4% macro-F1).
- **Slot/CTC head is weak** even on validation: only 17.3% exact-match, WER 0.387. The
  rule-based `parse_slots` is only as good as the decoded transcript, so **slot reliability
  is the model's real weakness** — independent of the Pi and of int8.
- `stop_timer` was never represented in validation → untested.

**Therefore the ≥ 90% intent gate is met on-distribution, but real-world performance on
recorded human speech is unproven.** The decisive next step (Area 1's open item) is a
**small real recorded dataset** (a few dozen clips per intent, including `stop_timer`
and OOV), then retrain and re-validate — expecting the slot head to improve most.

## 6. Bugs found & fixed in this pass

1. `inference/infer.py` — `resolve_checkpoint()` referenced undefined `HERE`
   (leftover from the Area-2 rename to `ROOT`) → `NameError` on any real run. Fixed.
2. `optimized_model.onnx` — shape-inference conflict broke `quantize_dynamic`. Resolved by
   re-exporting a clean dynamic graph from `best.pt` before quantizing.

## 7. Reproduce

```bash
# build the deployment model (Mac)
venv/bin/python model/export_onnx.py --checkpoint inference/best.pt --output vcm_model.onnx --opset 13
venv/bin/python model/quantize_onnx.py --input vcm_model.onnx --output vcm_model_int8.onnx --verify

# latency/RTF on the Pi 5
scp vcm_model_int8.onnx inference/benchmark.py config.py pi:~/vcm_bench/
ssh pi 'cd ~/vcm_bench && python3 benchmark.py --model vcm_model_int8.onnx --threads 2 --n-runs 50 --warmup 10'

# thermal soak
ssh pi 'cd ~/vcm_bench && python3 benchmark.py --model vcm_model_int8.onnx --threads 2 --soak 60'

# quality on a labeled set (labels.csv: path,intent[,transcript])
ssh pi 'cd ~/vcm_bench && python3 benchmark.py --model vcm_model_int8.onnx --threads 2 --wav-dir bench_wavs'
```

---

## 8. Pi 5 Model Benchmark: v6 vs v1 (2026-10-01)

**Date:** 2026-10-01  
**Target device:** Raspberry Pi 5 (Broadcom BCM2712 Cortex-A76 x4 @ 2.4 GHz, Linux 6.18 aarch64, glibc 2.41, ONNX Runtime 1.30.0, NumPy 2.2.4, Python 3.13.5)  
**Configuration:** `threads=2`, `n_runs=50`, `warmup=10`, durations = 0.5s, 1.0s, 1.5s, 2.0s (true-length mel inputs).  
**Reports:** Saved under `benchmark-results/` on both local machine and Pi 5.

### 8.1 Model Specifications

| Model | Variant | Task / Architecture | Size | SHA256 | Peak RSS |
|---|---|---|---|---|---|
| **v6_15m** | INT8 | Intent (18 classes) + 8 bounded numeric slot heads | 3.50 MB | `767da9c0717231c397fff7320d683fb79269b5b88e1ac403cd4e7f91b957584f` | 95.7 MB |
| **v6_15m** | FP32 | Intent (18 classes) + 8 bounded numeric slot heads | 7.95 MB | `8ec820a9a6b38b8b6c9398b6cedc161384c023822545230ae534be38c65c0a1f` | 86.3 MB |
| **v1** | INT8 | Intent classification only (18 classes) | 3.42 MB | `5cf0f5bf8b0bfbf8ad887f347c2e3313335c8eeb6bbc70cd19b3e1c8d9631427` | 95.8 MB |
| **v1** | FP32 | Intent classification only (18 classes) | 7.67 MB | `fe112e054e01472c703fe8325e5d140c420635126311e8e60cf3124243ce1e99` | 85.6 MB |

### 8.2 Latency and RTF on Pi 5 (2 Threads)

#### `v6_15m` (INT8) — [benchmark-results/pi5-v6-int8-t2.json](benchmark-results/pi5-v6-int8-t2.json)
| Audio Dur (s) | Frames | Mean (ms) | p50 (ms) | p95 (ms) | RTF | Status |
|---|---|---|---|---|---|---|
| 0.5 | 50 | 4.758 | 4.753 | 4.841 | 0.0095 | ✅ Pass |
| 1.0 | 100 | 11.222 | 10.986 | 12.209 | 0.0112 | ✅ Pass |
| 1.5 | 150 | 18.075 | 18.047 | 18.253 | 0.0121 | ✅ Pass |
| 2.0 | 200 | 24.933 | 24.911 | 25.137 | 0.0125 | ✅ Pass |

#### `v6_15m` (FP32) — [benchmark-results/pi5-v6-fp32-t2.json](benchmark-results/pi5-v6-fp32-t2.json)
| Audio Dur (s) | Frames | Mean (ms) | p50 (ms) | p95 (ms) | RTF | Status |
|---|---|---|---|---|---|---|
| 0.5 | 50 | 7.208 | 7.202 | 7.289 | 0.0144 | ✅ Pass |
| 1.0 | 100 | 13.928 | 13.721 | 14.530 | 0.0139 | ✅ Pass |
| 1.5 | 150 | 20.531 | 20.528 | 20.627 | 0.0137 | ✅ Pass |
| 2.0 | 200 | 27.464 | 27.445 | 27.577 | 0.0137 | ✅ Pass |

#### `v1` (INT8) — [benchmark-results/pi5-v1-int8-t2.json](benchmark-results/pi5-v1-int8-t2.json)
| Audio Dur (s) | Frames | Mean (ms) | p50 (ms) | p95 (ms) | RTF | Status |
|---|---|---|---|---|---|---|
| 0.5 | 50 | 4.788 | 4.784 | 4.840 | 0.0096 | ✅ Pass |
| 1.0 | 100 | 11.166 | 11.147 | 11.276 | 0.0112 | ✅ Pass |
| 1.5 | 150 | 18.936 | 18.630 | 19.128 | 0.0126 | ✅ Pass |
| 2.0 | 200 | 25.917 | 25.750 | 26.020 | 0.0130 | ✅ Pass |

#### `v1` (FP32) — [benchmark-results/pi5-v1-fp32-t2.json](benchmark-results/pi5-v1-fp32-t2.json)
| Audio Dur (s) | Frames | Mean (ms) | p50 (ms) | p95 (ms) | RTF | Status |
|---|---|---|---|---|---|---|
| 0.5 | 50 | 7.206 | 7.026 | 8.055 | 0.0144 | ✅ Pass |
| 1.0 | 100 | 13.533 | 13.540 | 13.615 | 0.0135 | ✅ Pass |
| 1.5 | 150 | 20.433 | 20.270 | 20.406 | 0.0136 | ✅ Pass |
| 2.0 | 200 | 27.364 | 27.194 | 27.551 | 0.0137 | ✅ Pass |

### 8.3 Acceptance Findings

1. **RTF Threshold (<= 0.3):**
   - All models achieve RTF between **0.0095 and 0.0144** on the Pi 5.
   - Inference runs approximately **70x to 105x faster than real-time audio**.
2. **p95 Latency Threshold (< 100 ms):**
   - For 1.0 s commands: **11.3–14.5 ms**.
   - For 2.0 s commands: **25.1–27.6 ms**.
   - Clears the 100 ms latency requirement with a ~4x safety margin.
3. **v6 Multi-Head vs v1 Intent-Only Overhead:**
   - Evaluating the 8 additional bounded slot classification heads in `v6_15m` introduces **zero measurable latency overhead** on the Pi 5 (24.9 ms for v6_int8 vs 25.9 ms for v1_int8 at 2.0 s).
4. **Quantization Benefit:**
   - INT8 dynamic quantization cuts disk usage by **56%** (3.4–3.5 MB vs 7.7–8.0 MB).
   - Lowers 0.5 s utterance latency from ~7.2 ms to ~4.7 ms.
   - Peak RSS is ~96 MB across INT8 models, well within the Pi 5 memory profile.

### 8.4 Quality Evaluation Across Datasets

#### 1. Held-Out Human Speech Benchmark Dataset (`test.jsonl`, 17,154 real speech recordings)
Evaluated on the held-out test split of `training_package_capped`:

| Model | Variant | Test Samples | Intent Accuracy | Intent Macro-F1 | Notes |
|---|---|---|---|---|---|
| **v1** | INT8 | 17,154 (Full) | **97.77%** | **95.74%** | Full test evaluation (Loss: 0.0766) |
| **v1** | INT8 | 500 (Slice) | **97.20%** | — | Evaluated via ONNX Runtime CPU |
| **v6_15m** | FP32 | 500 (Slice) | **96.60%** | — | Multi-head with 8 bounded slot heads |
| **v6_15m** | INT8 | 500 (Slice) | **96.40%** | — | Quantization degradation is only 0.2% (1 sample diff) |

#### 2. Synthetic Smoke Benchmark Dataset (`bench_wavs/`, 48 formant-tone WAVs)
Evaluated directly on the Pi 5 via `inference/benchmark.py`:

| Model | Variant | Labeled Samples | Intent Accuracy | Mean Latency (ms) | p95 Latency (ms) |
|---|---|---|---|---|---|
| **v6_15m** | INT8 | 48 | 4.17% (2/48) | 12.89 ms | 17.26 ms |
| **v6_15m** | FP32 | 48 | 4.17% (2/48) | 15.52 ms | 20.56 ms |
| **v1** | INT8 | 48 | 6.25% (3/48) | 12.49 ms | 16.88 ms |
| **v1** | FP32 | 48 | 6.25% (3/48) | 15.05 ms | 19.95 ms |

*Note:* `bench_wavs` consists of artificial formant-tone sweeps generated by a toy sine-wave synthesizer (not human speech). As documented in Section 5, it serves as a pipeline/wiring smoke test rather than an accuracy benchmark for models trained on real human speech. Real speech accuracy is confirmed on `test.jsonl` above.
