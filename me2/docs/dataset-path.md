# Dataset source

- **Canonical dataset:** [airimonda/ai231-me2-voice-commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands)
- **Pinned revision:** `bfca0bbebfe17e00d6a685d1da573615d29ecdce`
- **Files used by the downloader:** `data/train-*.parquet`,
  `data/numerals-*.parquet`,
  `synthetic_negatives/train-*.parquet`, `synthetic_negatives/tune_oos-*.parquet`,
  and `variations.csv`. The tune set is kept separate from training manifests.
- **Download command:** from `me2/pi5-vcm/`, run `uv run pi5-vcm-download-data`
- **Preparation command:** `uv run pi5-vcm-prepare-data --dataset-dir <snapshot-path>`
  from `me2/pi5-vcm/`

The revision pin makes the source repeatable. The local Hugging Face cache path is
machine-specific and is not the dataset source of record. Preparation writes
speaker-disjoint `train.jsonl` and `validation.jsonl` manifests plus
`tune_oos.jsonl`, `numerals.jsonl`, and `split_manifest.json`, including
provenance-based synthetic-negative counts.
