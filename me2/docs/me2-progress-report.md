# ME2 Progress Report

**Date:** 2026-10-14
**Scope:** Dataset source, preparation, and dataloaders in `pi5-vcm`

## Summary

The dataset input path has been implemented in `me2/pi5-vcm`. It can download a
revision-pinned subset of the public Hugging Face dataset, prepare split manifests,
and load labeled or unlabeled waveform batches from Parquet. The pinned snapshot
was present in the local Hugging Face cache and successfully prepared; generated
manifests remain local under the ignored `pi5-vcm/data/processed/` directory.

## Completed

- Added a Hugging Face downloader pinned to revision
  `bfca0bbebfe17e00d6a685d1da573615d29ecdce`.
- Added a preparation command that writes:
  - `train.jsonl`
  - `validation.jsonl`
  - `tune_oos.jsonl`
  - `numerals.jsonl`
  - `split_manifest.json`
- Ran preparation against the cached pinned snapshot at revision
  `bfca0bbebfe17e00d6a685d1da573615d29ecdce`.
- Implemented the planned deterministic, speaker-grouped greedy validation split
  (seed 42, target 10%, with validation coverage of each source-present class).
- Derived the 20 intent labels and six slot-head targets from dataset labels and
  `variations.csv`. Unsupported slot values receive target `-1` and mask `0`.
- Added provenance checks for synthetic training negatives. A negative is included
  only when every listed source resolves to an original training row outside the
  validation speakers. Missing, unresolved, ambiguous, or validation-linked
  provenance is excluded and counted.
- The 11 original train rows without `speaker_id` are conservatively treated as one
  unknown-speaker group. Provenance-linked negatives involving rows with unknown
  speakers are excluded.
- Kept `tune_oos` separate from the supervised manifests.
- Added a lazy Parquet-backed labeled audio loader with padded batches, lengths,
  masks, intent labels, and slot labels.
- Added an unlabeled numerals loader with checks for 16 kHz audio, mono channels,
  duration greater than 0.5 seconds, and finite samples.
- Documented the download and preparation workflow in
  [`pi5-vcm/README.md`](../pi5-vcm/README.md).

## Validation performed

- Nine fixture-based unit tests pass. Coverage includes deterministic splitting,
  speaker separation, class coverage, provenance filtering, unsupported-slot
  masking, JSONL-to-DataLoader integration, and numerals input validation.
- Python modules compile successfully.
- The preparation CLI help command works.
- The `uv` lockfile check passes.
- The cached dataset contains 10,733 original train rows, 66,390 numerals rows,
  1,000 synthetic train-negative rows, and 400 `tune_oos` rows.
- Preparation produced 1,074 validation rows (10.0065% of original train), 9,659
  original training rows, and 394 provenance-safe synthetic negatives. Validation
  covers all 20 intent/OOS classes. Synthetic-negative exclusions: 606 total
  (200 missing provenance, 307 unresolved provenance, 97 validation-linked, and
  2 referencing rows without speaker IDs).
- Decoded one real Parquet audio example from each generated manifest
  (`train`, `validation`, `tune_oos`, and `numerals`) successfully at 16 kHz.
- Completed a full decode/integrity pass across all 10,053 training-manifest rows,
  1,074 validation rows, 400 `tune_oos` rows, and 66,390 numerals rows. All passed
  sample-rate, channel-count, and finite-sample checks except that 100 numerals
  clips are shorter than the plan's required 0.5 seconds.
- **Phase 1 pretraining is skipped for this snapshot**, following the plan's
  explicit malformed-corpus fallback. Use random initialization for Phase 2.
  The numerals loader enforces the duration check and will reject those 100 clips;
  it does not silently filter or train on them.
- Split manifest SHA-256:
  `2891e8fd1af3d5305bc0f720084a2ee85af050b01224f01121d8295b8e3c4c0a`.

## Not yet verified or completed

- Phase 1 requires a different numerals snapshot or a plan-approved change to its
  duration requirement; neither has been made.
- `tune_oos` remains a separate manifest. Threshold tuning has not been run.
- The loader currently reads waveforms. Log-mel extraction, normalization
  statistics (`mel_stats.npz`), and masked-frame construction belong to the
  planned feature/pretraining implementation and are not part of this completed
  data-loader scope.
- Model, loss, training, export, inference, benchmark, and deployment work remains
  outside this report's scope.

## Reproduction commands

Run from `me2/pi5-vcm/`:

```bash
uv run pi5-vcm-download-data
uv run pi5-vcm-prepare-data \
  --dataset-dir <snapshot-path-printed-by-downloader> \
  --output-dir data/processed
```

The downloader and preparation details are recorded in
[`dataset-path.md`](./dataset-path.md). Generated data and the project virtual
environment are excluded from version control by `pi5-vcm/.gitignore`.
