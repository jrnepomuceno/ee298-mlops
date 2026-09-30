# New Training Track

This directory contains a training-only path for the read-only dataset at `../../Datasets_v2/training_package_capped`. It produces a package-specific intent classifier and INT8 ONNX artifact; it does not modify the reference package or the existing Raspberry Pi code/model.

## Current VCM Architecture

The existing target is `model/model.py::VCM`, with baseline defaults supplied by `model/main.py`:

```text
Input: 16 kHz mono audio
  -> Kaldi fbank log-mel features: (T, 80), 25 ms window, 10 ms hop
  -> Conv2d(1, 64, 3x3) + BatchNorm + ReLU
  -> MaxPool(time x2, frequency unchanged)
  -> Conv2d(64, 128, 3x3) + BatchNorm + ReLU
  -> MaxPool(time x2, frequency unchanged)
  -> reshape each time step: 128 x 80 = 10,240 features
  -> Linear(10,240, 128)
  -> 2-layer bidirectional GRU, hidden size 128 per direction
  -> 256-wide sequence representation
       -> masked mean pool -> 256 -> 128 -> intent logits
       -> per-frame projection -> 256 -> 128 -> CTC vocabulary logits
```

The temporal reduction is x4. The baseline has 64 convolution channels at the first stage and 128 at the second; the configurable `large` preset uses 96 then 192 channels and hidden size 192. The current default intent taxonomy is 18 commands plus `oov` (19 output classes). The CTC head emits the constrained vocabulary in `config.py`; it is not open-vocabulary speech recognition. Training uses intent cross-entropy plus CTC, with CTC disabled for OOV rows and true lengths used for pooling/loss.

This is a multi-task command-plus-transcript architecture, not yet a direct match for the capped package. Treat it as the target architecture to evaluate, not as a drop-in compatible model. The training model's intent ordering and output meaning must be explicit in its checkpoint metadata. The package's 18 labels are a different taxonomy, and this package has no OOV class/examples. Do not assume the existing Pi command executor or parser can consume the new head.

## Training Versus INT8 ONNX

The dataloader is only the input side of training. Its output is a trained PyTorch checkpoint, not an ONNX file. The repository's deployment path is a separate sequence: checkpoint -> FP32 ONNX export -> optional graph optimization -> ONNX Runtime dynamic INT8 weight quantization. Current export inputs are float32 mels `(B, T, 80)` and outputs are `intent_logits` and `ctc_logits`; dynamic quantization changes eligible weights to INT8 while activations remain float32. The existing quantizer checks the configured file-size budget and its optional benchmark is a host CPU proxy, not Pi validation.

The dataset-specific model must be frozen before export: output names/shapes, intent order, CTC vocabulary, feature parameters, and checkpoint reconstruction metadata must all match. In particular, changing the intent head to the package's 18 labels without updating the exporter and inference consumer would produce an ONNX artifact with different output semantics from the current deployed model. INT8 quantization cannot repair that incompatibility. Export and quantization are downstream acceptance gates, not dataloader steps; no export, transfer, or deployment is authorized/performed by this training-only documentation.

## Reference Package Audit

Read-only source: `../../Datasets_v2/training_package_capped`.

| Split file | Rows | `split` value in rows |
|---|---:|---|
| `train.jsonl` | 48,358 | `train` |
| `validation.jsonl` | 13,875 | `validation` |
| `test.jsonl` | 17,154 | `test` |
| **Total** | **79,387** | |

Every referenced audio file exists. All examples are 16 kHz, mono. Package fields include `audio_path`, `intent_label`, `transcript`, `speaker_id`, `split`, and `duration_seconds`; audio paths are package-relative. The package has 18 intent labels: `ALARM`, `BRIGHTNESS`, `CALL`, `CREATE_REMINDER`, `LIGHT_OFF`, `LIGHT_ON`, `LIST_REMINDERS`, `MESSAGE`, `NEXT`, `PAUSE`, `PLAY_MUSIC`, `STOP`, `TEMPERATURE`, `TIME`, `TIMER`, `VOLUME_DOWN`, `VOLUME_UP`, and `WEATHER`.

### Data -> Dataset/Loader contract

The current `VCMDataset` expects real rows with `path` and `intent`; `make_dataloader` and `collate_fn` then emit padded mel batches, integer intents, transcripts, flattened CTC token IDs, token lengths, and true mel-frame lengths. The package uses `audio_path` and `intent_label`, so direct loading fails or silently maps labels to the current OOV ID after a naive field rename.

The package manifests already define the official partition. Normalize `audio_path` to an absolute `path` rooted at the package directory, normalize `intent_label` to a deliberately defined training label, and change only the split spelling `validation` to `val`. Keep the speaker and sample identifiers for auditing. Pass the same normalized manifest and seed to both preprocessing and training. The existing `split_samples` preserves recognized split membership and shuffles within each split; it recognizes `val`, not `validation`. Without normalization it falls back to a random 80/10/10 split, which discards the intended partition and risks leakage.

Do not use the package-relative paths as-is from an arbitrary working directory. Do not merge the three files and independently re-split them. Keep the original train/validation/test membership intact.

### Dataset -> VCM contract

The current classifier taxonomy is different from the package's taxonomy. There is no safe one-to-one mapping for several labels: examples include `NEXT`, `MESSAGE`, and `BRIGHTNESS`, which have no corresponding current intent; conversely the current intents `mute`, `stop_timer`, and `dim_lights` have no matching package class. The package also has no OOV examples. Define a new training-only label order (the 18 package intents, and a separately sourced OOV class if rejection behavior is required) and size/reinitialize the intent head to match. Do not map unknown package labels to OOV or merge semantically distinct labels just to make dimensions fit.

The existing CTC tokenizer does not cover the package transcripts: in an audit of all rows, 119,672 of 402,251 whitespace-tokenized transcript tokens (about 29.7%) were absent from the current CTC vocabulary and become `<unk>`. Add and validate an appropriate vocabulary/tokenization policy before using CTC, or train an intent-only head initially and add a separately specified slot target strategy. The package provides `slot_values`; that structured field may be a better source of slot supervision than asking this constrained CTC head to reproduce every word in the transcript. Report CTC coverage and invalid/too-short targets before any training run.

### Duration and feature truncation

Audio duration ranges from 0.4 to 18.685 seconds. With 100 mel frames/second and the current default `max_frames=400`, approximately 4 seconds are retained. 2,073 training, 246 validation, and 277 test recordings exceed 4 seconds. The current precompute script also hardcodes `MAX_FRAMES = 400`, independent of the CLI/model setting. Establish an explicit policy before precompute: inspect long examples and transcript/slot alignment, then either raise the cap or intentionally truncate and record the resulting coverage. Do not silently truncate slot-bearing utterances.

Precomputed rows must remain aligned with the normalized manifest order and must include true, pre-padding lengths. The current implementation's fingerprint, sidecars, and per-worker mmap opening are useful safeguards; preserve equivalent checks in the training-only path.

## Dataset Rights

The package is suitable for local development under its source terms, not automatically unrestricted redistribution. SLURP audio is CC BY-NC 4.0, Option B permissions are unverified, and the synthetic eSpeak audio has no dataset-level license notice in this workspace. Keep training artifacts private to the authorized environment unless each source's terms have been reviewed. See the package's `LICENSES_AND_PROVENANCE.md`.

## Daniel-PC Runbook

The implementation in this directory is self-contained apart from installed PyTorch/torchaudio. The validated Daniel-PC interpreter is Python 3.12 with PyTorch/torchaudio 2.6.0+cu124 and a working RTX 3060 Ti. Run commands from `C:\Users\jdrne\TrainingData`, with this directory present as `new_training`. The capped package archive is already staged there and its SHA256 matches the accompanying checksum file.

First verify dependencies without replacing the existing CUDA build:

```powershell
py -3.12 -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__, torch.cuda.is_available())"
py -3.12 -m pip install -r new_training\requirements.txt
```

Extract the already-verified archive once, then validate the data path and precompute deterministic features. The default keeps all audio frames; no 4-second crop is applied. The feature files are variable-length FP16 rows in flat per-split stores, with true lengths, offsets, and fingerprint sidecars:

```powershell
New-Item -ItemType Directory -Force datasets | Out-Null
tar -xf training_package_capped.tar.zst -C datasets
py -3.12 -m new_training.smoke_test --package-root datasets\training_package_capped
py -3.12 -m new_training.precompute_mels --package-root datasets\training_package_capped --out-dir new_training\features --workers 4 --dtype float16
```

Start with a two-batch CUDA smoke run, then launch the full early-stopped training run:

```powershell
py -3.12 -m new_training.train_intent --package-root datasets\training_package_capped --features-dir new_training\features --output-dir new_training\runs\intent_v1 --device cuda --batch-size 32 --num-workers 4 --epochs 30 --patience 7 --max-batches 2
py -3.12 -m new_training.train_intent --package-root datasets\training_package_capped --features-dir new_training\features --output-dir new_training\runs\intent_v1 --device cuda --batch-size 32 --num-workers 4 --epochs 30 --patience 7
```

The full run writes `best.pt` and `history.json` under the output directory. After reviewing validation/test metrics, export and check the new intent-only INT8 ONNX artifact:

```powershell
py -3.12 -m new_training.export_int8 --checkpoint new_training\runs\intent_v1\best.pt --fp32-output new_training\runs\intent_v1\model_fp32.onnx --int8-output new_training\runs\intent_v1\model_int8.onnx --verify
```

The ONNX contract is one float32 `(B, T, 80)` input named `mels` and one 18-logit output named `intent_logits`. This artifact is not compatible with the existing dual-head Pi model without a separately reviewed inference/deployment change. Do not transfer it to the Pi as part of this run.

## Further Reading

- [Dataloader and 3060 Ti plan](DATALOADER_AND_PERFORMANCE.md)
- Existing implementation to compare: `model/dataset.py`, `model/dataloader.py`, `model/precompute_mels.py`, `model/model.py`, and `model/train.py`
