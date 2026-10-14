# pi5-vcm

## Reproducible dataset source

The canonical source is the public [ME2 voice-command dataset on Hugging Face](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands),
repository `airimonda/ai231-me2-voice-commands`. Downloads are pinned to revision
`bfca0bbebfe17e00d6a685d1da573615d29ecdce`. The downloader fetches the original
training and numerals Parquet shards, synthetic training negatives, the separate
`tune_oos` evaluation set, and `variations.csv`. `tune_oos` is never included in
training.

```bash
uv run pi5-vcm-download-data
```

The command prints the local Hugging Face snapshot directory. To use a non-default
Hugging Face cache location:

```bash
uv run pi5-vcm-download-data --cache-dir /path/to/huggingface-cache
```

The download is approximately 2.9 GB, primarily because the Phase 1 numerals corpus
is large. Test and holdout files are not downloaded by this command.

## Prepare training and validation manifests

After downloading, build the plan's speaker-grouped split and provenance-filtered
training manifests:

```bash
uv run pi5-vcm-prepare-data \
  --dataset-dir /path/printed/by/pi5-vcm-download-data \
  --output-dir data/processed
```

The command writes `train.jsonl`, `validation.jsonl`, `tune_oos.jsonl`,
`numerals.jsonl`, and `split_manifest.json`. `tune_oos.jsonl` is a separate
evaluation manifest for the planned τ threshold tuning; it is not merged into
train or validation. `numerals.jsonl` contains unlabeled Phase 1 audio rows.
Manifest Parquet references are relative to the downloaded snapshot, so the
snapshot path can change between machines without changing the manifests or their
hashes; pass the downloaded snapshot as `dataset_dir` when creating a loader.
The original training rows alone determine the 10% validation split; the
speaker-grouped greedy algorithm uses seed 42 and requires validation coverage of
every class present in the source train set. Synthetic training negatives are added
only when every `source_files` entry resolves to an original training row assigned
to the training portion. Original rows missing `speaker_id` are conservatively
kept together as one unknown-speaker group; negatives referring to them are
excluded because their speaker provenance cannot be established. Exclusion counts
and reasons, class counts, row IDs, speaker groups, manifest hashes, and the
split-manifest hash are recorded. The
`tune_oos` data remains outside the supervised manifests for its separate
threshold-tuning role.

## Parquet audio loader

The training data loader reads a JSONL manifest that indexes audio rows in the
dataset's Parquet files. Each line must contain:

```json
{
  "clip_id": "unique-clip-id",
  "parquet_path": "data/train-00000-of-00002.parquet",
  "row_group": 0,
  "row_index": 0,
  "intent_id": 0,
  "slot_targets": [-1, -1, -1, -1, -1, -1],
  "slot_mask": [0, 0, 0, 0, 0, 0],
  "speaker_id": "speaker-1"
}
```

Prepared manifests store paths relative to the dataset snapshot. Pass that snapshot
directory as `dataset_dir` to the loader. Handwritten manifests may instead use
paths relative to the manifest when `dataset_dir` is omitted. Intent IDs use the
20-class model contract, with out-of-scope as class 19. The six slot targets are
in model-head order and use `-1`/mask `0` for unsupported or inapplicable values.
The loader decodes waveform audio lazily and requires the configured 16 kHz sample
rate.

```python
from pi5_vcm.dataset import create_dataloader

train_loader = create_dataloader(
    "data/processed/train.jsonl",
    batch_size=32,
    shuffle=True,
    num_workers=2,
    dataset_dir="/path/printed/by/pi5-vcm-download-data",
)
for batch in train_loader:
    waveforms = batch["audio"]          # [batch, padded_samples]
    lengths = batch["audio_lengths"]    # [batch]
    intents = batch["intent_targets"]   # [batch]
    slots = batch["slot_targets"]      # [batch, 6], -1 means ignored
    slot_mask = batch["slot_mask"]      # [batch, 6]
```

For Phase 1, `ParquetNumeralsDataset` and `create_numerals_dataloader` enforce the
plan's 16 kHz, mono, finite-audio, and greater-than-0.5-second requirements:

```python
from pi5_vcm.dataset import create_numerals_dataloader

numerals_loader = create_numerals_dataloader(
    "data/processed/numerals.jsonl",
    batch_size=32,
    dataset_dir="/path/printed/by/pi5-vcm-download-data",
)
```

The pinned snapshot was checked across all 66,390 numerals rows: all decode as
16 kHz mono finite audio, but 100 clips are shorter than 0.5 seconds. Per the plan's
fallback, treat Phase 1 pretraining as unavailable for this snapshot and train
Phase 2 from random initialization. The numerals loader rejects those short clips
rather than silently admitting them into Phase 1.

## ME2 model

`pi5_vcm.model.ME2Model` implements the plan's 20-intent and six-slot output
contract. It accepts normalized 80-bin log-mel features with shape
`[batch, frames, 80]` and an optional valid-frame mask, and returns intent logits
`[batch, 20]` and slot logits `[batch, 6, 3]`. Its 256-channel causal DS-CNN
produces frame-aligned features and has approximately one million parameters,
including the output heads. The encoder can be frozen for Stage A and unfrozen
for Stage B with `set_encoder_trainable`.

`pi5_vcm.loss.multitask_loss` computes intent cross-entropy plus the six masked
slot losses; each slot loss is restricted to examples of its corresponding
intent. Use `inverse_frequency_class_weights` to construct the Stage A/B
class-balanced intent weights. The waveform loader above is not a log-mel
frontend: waveform-to-feature extraction and the training loop are separate
steps and must supply these model inputs.
