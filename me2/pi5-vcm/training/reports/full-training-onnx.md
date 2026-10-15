# Full Phase 2 Training and ONNX Candidate Results

## Run

- Phase 1 was skipped as allowed by the plan: 100 numerals clips in the pinned
  snapshot are shorter than the required 0.5 seconds.
- Phase 2 trained from random initialization on the complete training manifest:
  10,053 examples. The 1,074-example grouped validation manifest was used for
  early stopping and candidate evaluation.
- Hardware/software: NVIDIA RTX 3060 Ti (8 GiB), Windows 11, Python 3.12.10,
  PyTorch 2.6.0+cu124, CUDA 12.4.
- Reproducibility: seed 42, deterministic algorithms enabled, CUDA FP16,
  batch size 32.
- Training completed all planned epochs: Stage A 3/3 and Stage B 5/5. The best
  checkpoint is Stage B epoch 4, with validation intent Macro-F1 **0.41634**.
  The Stage B epoch 5 validation Macro-F1 was 0.39813.
- Dataset revision:
  `bfca0bbebfe17e00d6a685d1da573615d29ecdce`.
- Manifest SHA-256:
  - Train: `52fed9b996bbe480410e0ba3c03b6d8d64b6376162b62be7b050ca167848f8bb`
  - Validation: `a9401762818751523740f01f7eb8025c30660719f754509d7909b34804b36e9a`
  - `tune_oos`: `aac9c08113e102cdb863ce514697cc0cfc54d9478074fb063209a56e00466ba9`

Full epoch metrics and source/environment provenance are in
[`training_run.json`](../full-run-artifacts/training_run.json). The trained
checkpoint and its training-only mel statistics are alongside it.

## ONNX candidates

The exporter created FP32, FP16, and static QDQ INT8 candidates. The INT8
calibration used a deterministic 128-clip subset of the training manifest only.
All candidates were run through ONNX Runtime on the full validation and
`tune_oos` manifests. `test` and `holdout` were not accessed.

| Candidate | Size (bytes) | Intent argmax agreement vs FP32 | Mean abs. intent probability error | τ result | Candidate gates |
|---|---:|---:|---:|---|---|
| FP32 | 4,062,734 | 100% (reference) | 0 | τ=0.50; in-scope recall 37.59%; 160/400 OOS false accepts (40.00%) | Fails the ≥95% recall constraint |
| FP16 | 2,105,096 | 99.91% | 0.0000387 | τ=0.50; in-scope recall 37.59%; 160/400 OOS false accepts (40.00%) | Numerical and slot-accuracy gates pass; τ gate fails |
| INT8 | 1,331,415 | 85.66% | 0.010581 | τ=0.50; in-scope recall 34.12%; 137/400 OOS false accepts (34.25%) | Fails intent parity, probability-error, slot-accuracy, and τ gates |

No tested τ in the planned range `[0.50, 0.99]` reached 95% in-scope recall.
Per the plan, τ=0.50 is the fallback maximizing observed recall, and the
achieved recall and false-accept counts are reported above. The INT8 false-accept
rate is lower, but its recall and numerical/slot gates fail.

Candidate hashes, operator counts, calibration clip IDs/hash, per-slot
accuracies, and full metrics are in
[`export_report.json`](../onnx-candidates/export_report.json).

## Outcome and remaining gate

**No candidate is accepted for deployment.** FP16 passes conversion-parity and
slot-accuracy checks but cannot meet the base model's recall constraint. INT8
fails additional numerical and slot-accuracy checks. Therefore no
`model.onnx` was selected or promoted.

The plan's RPi4 provider/node profiling and end-to-end latency gate were not
run; these candidates are not certified for deployment. The next work should
address the supervised model's low in-scope recall before repeating export
acceptance. Do not evaluate `test` or `holdout` during that work.
