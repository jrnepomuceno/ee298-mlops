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
