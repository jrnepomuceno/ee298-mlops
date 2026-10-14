from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

DATASET_URL = "https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands"
DATASET_REPO_ID = "airimonda/ai231-me2-voice-commands"
DATASET_REVISION = "bfca0bbebfe17e00d6a685d1da573615d29ecdce"
DATASET_DOWNLOAD_PATTERNS = (
    "data/train-*.parquet",
    "data/numerals-*.parquet",
    "synthetic_negatives/train-*.parquet",
    "synthetic_negatives/tune_oos-*.parquet",
    "variations.csv",
)


def download_training_data(*, cache_dir: str | Path | None = None) -> Path:
    """Download only the training inputs from the pinned public dataset revision."""
    snapshot_path = snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
        allow_patterns=list(DATASET_DOWNLOAD_PATTERNS),
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return Path(snapshot_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the pinned ME2 training dataset files."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Hugging Face cache directory (defaults to the standard user cache).",
    )
    args = parser.parse_args()
    snapshot_path = download_training_data(cache_dir=args.cache_dir)
    print(f"Dataset revision: {DATASET_REVISION}")
    print(f"Training data snapshot: {snapshot_path}")


if __name__ == "__main__":
    main()
