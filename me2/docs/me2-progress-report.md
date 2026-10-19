# ME2 Progress Report

**Date:** 2026-10-19
**Scope:** Dataset pipeline and Phase 1 Pi inference runner in `pi5-vcm`

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

## Phase 1 inference runner update (2026-10-19)

### Completed

- Added a CPU ONNX Runtime predictor for the expected ME2 contract: float
  `mel_features`, boolean `frame_mask`, 20 intent probabilities, and six
  3-class slot heads. The frontend computes 80-bin log-mel features and applies
  paired training statistics.
- Added local action dispatch for time, optional GPIO light control, and
  `playerctl` media actions. Unsupported integrations report `unavailable`;
  out-of-scope predictions are rejected.
- Implemented current WEATHER responses for Quezon City, Philippines, using
  OpenWeather Current Weather API and `OPENWEATHER_API_KEY`. The provider
  requests metric conditions by city name and avoids logging the API key or
  URL. The systemd installer supports an optional private runtime environment
  file. The key was present in the Pi's `.bashrc` but was not inherited by
  systemd; it was copied into the private environment file and verified present
  in the running process without exposing its value. One Call API 4.0 returned
  HTTP 401 for that key, so the provider was switched to Current Weather API.
  The Current Weather API returned live Quezon City conditions in a spoken Pi
  interaction. Speech says the temperature in metric "degrees" without the word
  "Celsius", which this Piper voice mispronounced; the API units and numeric
  value remain metric.
- Extended the Pi 5 wake runner with optional command capture, inference or
  explicit test-intent dispatch, and Piper speech output. Persistent
  `--dummy-time` and `--dummy-intent INTENT` modes are opt-in and bypass
  inference; they do not use a fabricated ONNX model. The service installer can
  persist either test mode. The Pi service is currently set to persistent
  `--dummy-intent WEATHER`, which dispatches WEATHER for every voiced command
  after Alexa, regardless of the command's words.
- Loaded Piper's voice once and reused it for replies. Added silent startup
  warmup for the wake-word model and Piper synthesis.
- Added per-interaction timing logs for command capture, intent execution,
  Piper synthesis, speech playback, and total time from wake acknowledgement to
  completion.
- Added optional HyperX DuoCast RGB states using a foreground QuadcastRGB build:
  rainbow cycle during warmup, off in standby, blue on wake, green while the
  spoken reply plays, then off. The existing optional GPIO indicator remains
  available separately.
- Deployed the runner and QuadcastRGB foreground executable to the Raspberry
  Pi 5. The systemd user service was verified active in persistent WEATHER mode
  with DuoCast audio capture running.
- One live wake-to-spoken-WEATHER interaction completed successfully on the
  Pi. Its measured durations were: command capture 4,181 ms (includes the
  spoken command and 800 ms end-of-speech silence); WEATHER execution 367 ms;
  speech synthesis 637 ms; playback 5,309 ms; and 10,925 ms from wake
  acknowledgement through playback completion. These are one-interaction
  measurements, not latency percentiles. Playback duration measures the whole
  spoken response, not time to first audible output.
- Focused wake-runner and intent-execution tests: 58 passed. Python compilation,
  installer shell syntax, and `git diff --check` passed.

### Not yet verified or completed

- The ONNX models available on the Pi do not match the predictor's ME2 input,
  output, and slot contract. A compatible exported model and its paired
  `mel_stats.npz` are still required before real ME2 predictions can be run.
- The dispatcher currently returns `unavailable` for `CALL`, `MESSAGE`,
  `LIST_REMINDERS`, `TIMER`, `ALARM`, `TEMPERATURE`, `BRIGHTNESS`, `COLOR`, and
  `CREATE_REMINDER`; their service or hardware adapters are not configured.
  `OUT_OF_SCOPE` is rejected by design.
- `LIGHT_ON`/`LIGHT_OFF` require a configured GPIO pin, and media intents
  (`PLAY_MUSIC`, `PAUSE`, `STOP`, `NEXT`, `VOLUME_UP`, `VOLUME_DOWN`) require a
  working `playerctl` target. Their real-device behavior has not been validated
  on the Pi in this progress update.
- Persistent `--dummy-intent WEATHER` is a test/demo mode, not model-routed
  inference: it dispatches WEATHER after every voiced command regardless of
  what was said. Real ME2 inference remains blocked on a compatible exported
  model and its paired `mel_stats.npz`.
- One live WEATHER interaction has been timed, but latency percentiles and the
  Phase 1 benchmark requirements have not been measured.
- End-to-end latency percentiles and the Phase 1 benchmark requirements have
  not been measured.

## Original dataset-pipeline follow-ups

- Phase 1 requires a different numerals snapshot or a plan-approved change to its
  duration requirement; neither has been made.
- `tune_oos` remains a separate manifest. Threshold tuning has not been run.
- The loader currently reads waveforms. Log-mel extraction, normalization
  statistics (`mel_stats.npz`), and masked-frame construction belong to the
  planned feature/pretraining implementation and are not part of this completed
  data-loader scope.
- Model training, export, and benchmark work remain incomplete; the inference
  runner and deployment progress are recorded in the update above.

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
