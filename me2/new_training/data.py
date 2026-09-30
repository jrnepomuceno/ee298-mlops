"""Package adapter and length-aware DataLoader for the new intent model."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset

SPLIT_FILES = {
    "train": "train.jsonl",
    "val": "validation.jsonl",
    "test": "test.jsonl",
}
FEATURE_BINS = 80


def load_package(package_root: str | Path) -> tuple[list[str], dict[str, list[dict]]]:
    """Load fixed package splits and resolve package-relative audio paths."""
    root = Path(package_root).resolve()
    intents_path = root / "intents.json"
    if not intents_path.is_file():
        raise FileNotFoundError(intents_path)
    labels = [entry["label"] for entry in json.loads(intents_path.read_text(encoding="utf-8"))]
    if len(labels) != len(set(labels)):
        raise ValueError("intents.json contains duplicate labels")
    label_set = set(labels)

    splits: dict[str, list[dict]] = {}
    sample_ids: set[str] = set()
    for split, filename in SPLIT_FILES.items():
        manifest = root / filename
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        rows: list[dict] = []
        with manifest.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                source = json.loads(line)
                expected_split = "validation" if split == "val" else split
                if source.get("split") != expected_split:
                    raise ValueError(f"{filename}:{line_no}: unexpected split {source.get('split')!r}")
                label = source.get("intent_label")
                if label not in label_set:
                    raise ValueError(f"{filename}:{line_no}: unknown intent label {label!r}")
                sample_id = source.get("sample_id")
                if not sample_id or sample_id in sample_ids:
                    raise ValueError(f"{filename}:{line_no}: missing or duplicate sample_id {sample_id!r}")
                sample_ids.add(sample_id)
                relative_audio = Path(source["audio_path"])
                if relative_audio.is_absolute():
                    raise ValueError(f"{filename}:{line_no}: expected package-relative audio_path")
                audio_path = (root / relative_audio).resolve()
                if root not in audio_path.parents or not audio_path.is_file():
                    raise FileNotFoundError(f"{filename}:{line_no}: audio file missing or outside package: {audio_path}")
                row = dict(source)
                row["path"] = str(audio_path)
                row["intent"] = label
                row["normalized_split"] = split
                rows.append(row)
        splits[split] = rows
    return labels, splits


def split_fingerprint(rows: list[dict]) -> str:
    """Path-independent digest that also detects changes in row order."""
    digest = hashlib.sha256()
    for row in rows:
        stable = {
            "sample_id": row["sample_id"],
            "intent_label": row["intent_label"],
            "transcript": row["transcript"],
            "split": row["split"],
            "sha256": row["sha256"],
            "sample_rate": row["sample_rate"],
            "duration_seconds": row["duration_seconds"],
        }
        digest.update(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class MelFeatureDataset(Dataset):
    """Read variable-length feature rows from a split-level flat mmap."""

    def __init__(self, rows: list[dict], labels: list[str], features_dir: str | Path,
                 split: str, augment: bool = False):
        self.rows = rows
        self.label_to_id = {label: index for index, label in enumerate(labels)}
        self.features_dir = Path(features_dir)
        self.split = split
        self.augment = augment
        self._features: np.memmap | None = None

        metadata_path = self.features_dir / f"features_{split}.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (self.metadata.get("n_rows") != len(rows) or
                self.metadata.get("manifest_fingerprint") != split_fingerprint(rows) or
                self.metadata.get("n_mels") != FEATURE_BINS):
            raise ValueError(f"precomputed feature metadata mismatch for split {split}")

        self.offsets = np.load(self.features_dir / self.metadata["offsets_file"], allow_pickle=False)
        self.lengths = np.load(self.features_dir / self.metadata["lengths_file"], allow_pickle=False)
        if self.offsets.shape != (len(rows),) or self.lengths.shape != (len(rows),):
            raise ValueError(f"precomputed index shape mismatch for split {split}")
        if np.any(self.lengths < 1) or np.any(self.offsets < 0):
            raise ValueError(f"invalid precomputed offsets/lengths for split {split}")
        if len(rows) and (self.offsets[0] != 0 or
                          not np.array_equal(self.offsets[1:], self.offsets[:-1] + self.lengths[:-1])):
            raise ValueError(f"precomputed rows are not contiguous for split {split}")
        self.total_frames = int(self.lengths.sum())
        feature_path = self.features_dir / self.metadata["feature_file"]
        dtype = np.dtype(self.metadata["dtype"])
        expected_bytes = self.total_frames * FEATURE_BINS * dtype.itemsize
        if not feature_path.is_file() or feature_path.stat().st_size != expected_bytes:
            raise ValueError(f"precomputed feature file size mismatch for split {split}")
        self.feature_path = feature_path
        self.dtype = dtype
        self.label_ids = np.asarray([self.label_to_id[row["intent"]] for row in rows], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def close(self) -> None:
        if self._features is not None:
            self._features._mmap.close()
            self._features = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_features"] = None
        return state

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        if self._features is None:
            self._features = np.memmap(
                self.feature_path, mode="r", dtype=self.dtype,
                shape=(self.total_frames, FEATURE_BINS))
        start = int(self.offsets[index])
        length = int(self.lengths[index])
        mel = torch.from_numpy(np.asarray(self._features[start:start + length]).copy())
        if self.augment:
            mel = _spec_augment(mel)
        return mel, int(self.label_ids[index]), length


def _spec_augment(mel: torch.Tensor) -> torch.Tensor:
    if torch.rand(()) >= 0.5:
        return mel
    output = mel.clone()
    time_steps, frequency_bins = output.shape
    for _ in range(2):
        width = int(torch.randint(0, min(41, time_steps + 1), ()).item())
        if width:
            start = int(torch.randint(0, time_steps - width + 1, ()).item())
            output[start:start + width, :] = 0
        width = int(torch.randint(0, min(28, frequency_bins + 1), ()).item())
        if width:
            start = int(torch.randint(0, frequency_bins - width + 1, ()).item())
            output[:, start:start + width] = 0
    return output


def collate_intents(batch: list[tuple[torch.Tensor, int, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mels, labels, lengths = zip(*batch)
    max_length = max(lengths)
    dtype = mels[0].dtype
    padded = torch.zeros((len(mels), max_length, FEATURE_BINS), dtype=dtype)
    for index, mel in enumerate(mels):
        padded[index, :mel.shape[0]] = mel
    return padded, torch.tensor(labels, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


class LengthBucketBatchSampler(BatchSampler):
    """Shuffle locally, then batch similar lengths to reduce padding."""

    def __init__(self, lengths: np.ndarray, batch_size: int, shuffle: bool,
                 seed: int = 42, bucket_multiplier: int = 20):
        if batch_size < 1 or bucket_multiplier < 1:
            raise ValueError("batch_size and bucket_multiplier must be positive")
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_size = batch_size * bucket_multiplier
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        indices = np.arange(len(self.lengths))
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(indices)
        batches: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start:start + self.bucket_size]
            bucket = bucket[np.argsort(self.lengths[bucket], kind="stable")]
            batches.extend(bucket[i:i + self.batch_size].tolist()
                           for i in range(0, len(bucket), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)


def make_loader(dataset: MelFeatureDataset, batch_size: int, shuffle: bool,
                num_workers: int, pin_memory: bool, seed: int = 42) -> DataLoader:
    batch_sampler = LengthBucketBatchSampler(
        dataset.lengths, batch_size=batch_size, shuffle=shuffle, seed=seed)
    kwargs = {
        "dataset": dataset,
        "batch_sampler": batch_sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "collate_fn": collate_intents,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)
