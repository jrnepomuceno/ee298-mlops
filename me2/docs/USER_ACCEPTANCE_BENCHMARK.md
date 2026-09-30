# User Acceptance Benchmark: Raspberry Pi 5

**Status:** Prepared; not yet run on the target Pi.
**Purpose:** Evaluate the exact INT8 ONNX artifact and its WAV-to-intent/slot path on the Raspberry Pi 5. Model file size is recorded for reference but has no pass/fail limit for now.

## What Is Being Tested

The benchmark measures ONNX Runtime CPU inference at true mel length, feature extraction using the same NumPy Kaldi-fbank implementation as Pi inference, intent/slot quality against labels, process peak RSS, and Pi clock/temperature/throttle state. It records the model SHA256, file size, software versions, per-clip latency percentiles, and a JSON report.

`bench_wavs/` currently contains 48 labeled, synthetic formant-tone WAVs. They are useful for checking package wiring and repeatability, but they are not human speech and must not be presented as user acceptance quality evidence. Final UAT requires a held-out set of real recordings made by speakers who do not appear in training.

## Transfer Bundle

From the repository root, retain these paths and their relative layout:

- `config.py`
- `model/__init__.py`, `model/onnx_deploy.py`, `model/slots.py`
- `inference/benchmark.py`, `inference/features.py`
- `models/onnx/<tag>/vcm_model_int8.onnx` and its `manifest.json`
- `bench_wavs/labels.csv` and `bench_wavs/*.wav` for smoke testing
- NumPy and ONNX Runtime installed in the Pi environment

The selected ONNX must expose the VCM contract: float32 input `(B, T, 80)`, 19 intent logits, and 88 CTC logits per frame. The current model resolver defaults to the newest version folder; pass `--model` explicitly for a recorded acceptance run so a later folder cannot silently change the tested artifact.

## Preflight

Run from the repository root on the Pi:

```bash
python3 -c 'import numpy, onnxruntime; print("numpy", numpy.__version__, "onnxruntime", onnxruntime.__version__)'
sha256sum models/onnx/v19-20260929-2/vcm_model_int8.onnx
python3 - <<'PY'
import onnxruntime as ort
path = "models/onnx/v19-20260929-2/vcm_model_int8.onnx"
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
print("inputs", [(x.name, x.shape, x.type) for x in session.get_inputs()])
print("outputs", [(x.name, x.shape, x.type) for x in session.get_outputs()])
PY
```

Confirm input/output names and shapes match the expected VCM contract before proceeding. Record the Pi model, RAM configuration, OS, power supply, and ambient conditions with the result.

## Stage A: Latency and Thermal Run

This uses deterministic random mels, isolates `session.run()`, and is not a quality test. Run once at one thread and once at two threads; retain both reports.

```bash
mkdir -p benchmark-results
python3 inference/benchmark.py \
  --model models/onnx/v19-20260929-2/vcm_model_int8.onnx \
  --threads 1 --durations 0.5,1.0,1.5,2.0 \
  --n-runs 50 --warmup 10 --soak 60 --json \
  --output benchmark-results/pi5-v19-20260929-2-synthetic-t1.json

python3 inference/benchmark.py \
  --model models/onnx/v19-20260929-2/vcm_model_int8.onnx \
  --threads 2 --durations 0.5,1.0,1.5,2.0 \
  --n-runs 50 --warmup 10 --soak 60 --json \
  --output benchmark-results/pi5-v19-20260929-2-synthetic-t2.json
```

Current performance acceptance bars from `docs/plan.md` are RTF <= 0.3 and p95 inference latency < 100 ms for 1-2 second commands. There is no model-size bar. Review pre/post-soak clock, temperature, and throttling; investigate any throttle or unexplained post-soak regression.

## Stage B: Labeled Audio Quality Run

Use the bundled 48 clips only as a smoke-level check. The benchmark requires full label coverage and uses the canonical Pi feature extractor and slot parser.

```bash
python3 inference/benchmark.py \
  --model models/onnx/v19-20260929-2/vcm_model_int8.onnx \
  --wav-dir bench_wavs --require-labels --threads 2 \
  --n-runs 20 --warmup 5 --json \
  --output benchmark-results/pi5-v19-20260929-2-synthetic-uat.json
```

For genuine UAT, provide a separate directory of held-out human-recorded WAVs and a headerless or headed `labels.csv` with `path,intent,transcript` columns. Keep the set disjoint from training by speaker and recording session. Include every supported intent, slot-bearing examples across values, and OOV examples; report per-intent support so missing coverage is visible. Run the same command with `--wav-dir /path/to/heldout_human_wavs`.

The JSON report includes overall intent accuracy, supported-class macro-F1, per-intent precision/recall/F1 and confusion matrix, slot F1/exact match, OOV recall, non-OOV false-reject rate, model-only latency, feature-plus-model latency, and per-clip timing. Existing project policy sets intent accuracy >= 0.90 as a pass bar. The project has not set numeric slot/OOV UAT thresholds; record those metrics and agree thresholds before treating them as pass/fail.

## Interpretation and Sign-Off

- Do not use a Mac or Windows result as Raspberry Pi performance evidence.
- Do not interpret synthetic `bench_wavs` quality as real-speech accuracy.
- Keep each JSON report with the exact model hash and do not overwrite earlier runs.
- If the model manifest lacks test metrics, run this held-out UAT rather than inferring quality from validation metrics.
- Review all clips with incorrect intents or slots, then have a user confirm command behavior through the Pi's dry-run harness before enabling any real device actions.
- Record the tested ONNX tag/hash, Pi configuration, benchmark command, metrics, failures, and sign-off date in `docs/VALIDATION.md` after the run.
