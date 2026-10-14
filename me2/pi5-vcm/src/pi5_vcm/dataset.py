from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

NUM_INTENTS = 20
NUM_SLOT_HEADS = 6
NUM_SLOT_CLASSES = 3
SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class AudioRecord:
    clip_id: str
    parquet_path: Path
    row_group: int
    row_index: int
    intent_id: int | None
    slot_targets: tuple[int, ...] | None
    slot_mask: tuple[int, ...] | None
    speaker_id: str | None = None

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        *,
        manifest_dir: Path,
        dataset_dir: Path | None,
        allow_unlabeled: bool,
    ) -> AudioRecord:
        parquet_path = Path(value["parquet_path"])
        if not parquet_path.is_absolute():
            parquet_path = (dataset_dir or manifest_dir) / parquet_path

        intent_id = value.get("intent_id")
        record = cls(
            clip_id=str(value["clip_id"]),
            parquet_path=parquet_path,
            row_group=int(value["row_group"]),
            row_index=int(value["row_index"]),
            intent_id=int(intent_id) if intent_id is not None else None,
            slot_targets=(
                tuple(int(item) for item in value["slot_targets"])
                if "slot_targets" in value
                else None
            ),
            slot_mask=(
                tuple(int(item) for item in value["slot_mask"])
                if "slot_mask" in value
                else None
            ),
            speaker_id=(
                str(value["speaker_id"])
                if value.get("speaker_id") is not None
                else None
            ),
        )
        record.validate(allow_unlabeled=allow_unlabeled)
        return record

    def validate(self, *, allow_unlabeled: bool) -> None:
        if not self.clip_id:
            raise ValueError("clip_id must not be empty")
        if self.row_group < 0 or self.row_index < 0:
            raise ValueError(f"{self.clip_id}: Parquet row coordinates must be nonnegative")
        if self.intent_id is None:
            if not allow_unlabeled:
                raise ValueError(f"{self.clip_id}: missing intent label")
            if self.slot_targets is not None or self.slot_mask is not None:
                raise ValueError(
                    f"{self.clip_id}: unlabeled records cannot contain slot labels"
                )
            return
        if not 0 <= self.intent_id < NUM_INTENTS:
            raise ValueError(
                f"{self.clip_id}: intent_id must be in [0, {NUM_INTENTS - 1}]"
            )
        if self.slot_targets is None or len(self.slot_targets) != NUM_SLOT_HEADS:
            raise ValueError(
                f"{self.clip_id}: expected {NUM_SLOT_HEADS} slot targets"
            )
        if self.slot_mask is None or len(self.slot_mask) != NUM_SLOT_HEADS:
            raise ValueError(f"{self.clip_id}: expected {NUM_SLOT_HEADS} slot masks")
        for target, mask in zip(self.slot_targets, self.slot_mask, strict=True):
            if mask not in (0, 1):
                raise ValueError(f"{self.clip_id}: slot masks must be 0 or 1")
            if target != -1 and not 0 <= target < NUM_SLOT_CLASSES:
                raise ValueError(
                    f"{self.clip_id}: slot targets must be -1 or in "
                    f"[0, {NUM_SLOT_CLASSES - 1}]"
                )
            if target == -1 and mask != 0:
                raise ValueError(
                    f"{self.clip_id}: an unsupported slot value must have mask 0"
                )


class ParquetAudioDataset(Dataset[dict[str, Any]]):
    """Load waveform rows addressed by a JSONL Parquet manifest.

    Each manifest line contains ``clip_id``, ``parquet_path``, ``row_group``,
    ``row_index``, ``intent_id``, ``slot_targets`` and ``slot_mask``. Audio is
    decoded on demand, so waveform payloads are not held in memory for the
    lifetime of the dataset.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        sample_rate: int = SAMPLE_RATE,
        row_group_cache_size: int = 1,
        dataset_dir: str | Path | None = None,
        allow_unlabeled: bool = False,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if row_group_cache_size < 1:
            raise ValueError("row_group_cache_size must be at least 1")

        self.manifest_path = Path(manifest_path).resolve()
        self.dataset_dir = Path(dataset_dir).resolve() if dataset_dir else None
        self.sample_rate = sample_rate
        self.row_group_cache_size = row_group_cache_size
        self.allow_unlabeled = allow_unlabeled
        self.records = self._load_manifest()
        self._parquet_files: dict[Path, pq.ParquetFile] = {}
        self._row_groups: OrderedDict[tuple[Path, int], Any] = OrderedDict()
        self._cache_pid = os.getpid()

    def _load_manifest(self) -> list[AudioRecord]:
        records: list[AudioRecord] = []
        with self.manifest_path.open(encoding="utf-8") as manifest:
            for line_number, line in enumerate(manifest, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("manifest entry must be a JSON object")
                    records.append(
                        AudioRecord.from_dict(
                            value,
                            manifest_dir=self.manifest_path.parent,
                            dataset_dir=self.dataset_dir,
                            allow_unlabeled=self.allow_unlabeled,
                        )
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Invalid manifest entry at line {line_number}: {exc}"
                    ) from exc
        if not records:
            raise ValueError(f"Manifest contains no records: {self.manifest_path}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        audio_row = self._read_audio_row(record)
        audio_bytes = audio_row.get("bytes")
        if not isinstance(audio_bytes, (bytes, bytearray, memoryview)):
            raise ValueError(f"{record.clip_id}: Parquet audio row has no byte payload")

        waveform, actual_sample_rate = sf.read(
            BytesIO(audio_bytes), dtype="float32", always_2d=True
        )
        if actual_sample_rate != self.sample_rate:
            raise ValueError(
                f"{record.clip_id}: expected {self.sample_rate} Hz audio, "
                f"got {actual_sample_rate} Hz"
            )
        mono_waveform = np.ascontiguousarray(waveform.mean(axis=1))
        sample: dict[str, Any] = {
            "audio": torch.from_numpy(mono_waveform),
            "clip_id": record.clip_id,
            "speaker_id": record.speaker_id,
            "sample_rate": actual_sample_rate,
            "duration_s": len(mono_waveform) / actual_sample_rate,
            "channel_count": waveform.shape[1],
        }
        if record.intent_id is not None:
            if record.slot_targets is None or record.slot_mask is None:
                raise ValueError(f"{record.clip_id}: incomplete intent slot labels")
            sample.update(
                {
                    "intent_target": torch.tensor(record.intent_id, dtype=torch.long),
                    "slot_targets": torch.tensor(record.slot_targets, dtype=torch.long),
                    "slot_mask": torch.tensor(record.slot_mask, dtype=torch.bool),
                }
            )
        return sample

    def _read_audio_row(self, record: AudioRecord) -> dict[str, Any]:
        current_pid = os.getpid()
        if current_pid != self._cache_pid:
            self._parquet_files = {}
            self._row_groups = OrderedDict()
            self._cache_pid = current_pid

        cache_key = (record.parquet_path, record.row_group)
        if cache_key not in self._row_groups:
            parquet_file = self._parquet_files.get(record.parquet_path)
            if parquet_file is None:
                parquet_file = pq.ParquetFile(record.parquet_path)
                self._parquet_files[record.parquet_path] = parquet_file
            if record.row_group >= parquet_file.num_row_groups:
                raise IndexError(
                    f"{record.clip_id}: row group {record.row_group} does not exist "
                    f"in {record.parquet_path}"
                )
            row_group = parquet_file.read_row_group(
                record.row_group, columns=["audio"]
            )
            self._row_groups[cache_key] = row_group.column("audio")
            while len(self._row_groups) > self.row_group_cache_size:
                self._row_groups.popitem(last=False)
        else:
            self._row_groups.move_to_end(cache_key)

        audio_column = self._row_groups[cache_key]
        if record.row_index >= len(audio_column):
            raise IndexError(
                f"{record.clip_id}: row index {record.row_index} is outside "
                f"row group {record.row_group}"
            )
        audio_row = audio_column[record.row_index].as_py()
        if not isinstance(audio_row, dict):
            raise ValueError(f"{record.clip_id}: Parquet audio value is not a struct")
        return audio_row

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_parquet_files"] = {}
        state["_row_groups"] = OrderedDict()
        state["_cache_pid"] = None
        return state


def collate_audio(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length waveforms and stack the intent and slot targets."""
    if not samples:
        raise ValueError("cannot collate an empty batch")

    sample_rates = {sample["sample_rate"] for sample in samples}
    if len(sample_rates) != 1:
        raise ValueError("all samples in a batch must have the same sample rate")

    lengths = torch.tensor([sample["audio"].numel() for sample in samples])
    padded_audio = torch.nn.utils.rnn.pad_sequence(
        [sample["audio"] for sample in samples],
        batch_first=True,
        padding_value=0.0,
    )
    positions = torch.arange(padded_audio.shape[1])
    audio_mask = positions.unsqueeze(0) < lengths.unsqueeze(1)

    return {
        "audio": padded_audio,
        "audio_lengths": lengths,
        "audio_mask": audio_mask,
        "intent_targets": torch.stack([sample["intent_target"] for sample in samples]),
        "slot_targets": torch.stack([sample["slot_targets"] for sample in samples]),
        "slot_mask": torch.stack([sample["slot_mask"] for sample in samples]),
        "clip_ids": [sample["clip_id"] for sample in samples],
        "speaker_ids": [sample["speaker_id"] for sample in samples],
        "sample_rate": sample_rates.pop(),
    }


class ParquetNumeralsDataset(ParquetAudioDataset):
    """Unlabeled numeral audio with the plan's data-integrity checks."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        sample_rate: int = SAMPLE_RATE,
        row_group_cache_size: int = 1,
        dataset_dir: str | Path | None = None,
    ) -> None:
        super().__init__(
            manifest_path,
            sample_rate=sample_rate,
            row_group_cache_size=row_group_cache_size,
            dataset_dir=dataset_dir,
            allow_unlabeled=True,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        audio = sample["audio"]
        if sample["channel_count"] != 1:
            raise ValueError(
                f"{sample['clip_id']}: numerals audio must be mono, "
                f"got {sample['channel_count']} channels"
            )
        if sample["duration_s"] <= 0.5:
            raise ValueError(
                f"{sample['clip_id']}: numerals audio must be longer than 0.5 seconds"
            )
        if not torch.isfinite(audio).all().item():
            raise ValueError(f"{sample['clip_id']}: numerals audio contains NaN or infinity")
        return sample


def collate_numerals_audio(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad unlabeled numeral waveforms into a batch."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    sample_rates = {sample["sample_rate"] for sample in samples}
    if len(sample_rates) != 1:
        raise ValueError("all samples in a batch must have the same sample rate")

    lengths = torch.tensor([sample["audio"].numel() for sample in samples])
    padded_audio = torch.nn.utils.rnn.pad_sequence(
        [sample["audio"] for sample in samples],
        batch_first=True,
        padding_value=0.0,
    )
    positions = torch.arange(padded_audio.shape[1])
    return {
        "audio": padded_audio,
        "audio_lengths": lengths,
        "audio_mask": positions.unsqueeze(0) < lengths.unsqueeze(1),
        "clip_ids": [sample["clip_id"] for sample in samples],
        "sample_rate": sample_rates.pop(),
    }


def create_dataloader(
    manifest_path: str | Path,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
    dataset_dir: str | Path | None = None,
) -> DataLoader[dict[str, Any]]:
    """Create a PyTorch DataLoader for a prepared JSONL manifest."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if num_workers < 0:
        raise ValueError("num_workers must not be negative")
    dataset = ParquetAudioDataset(manifest_path, dataset_dir=dataset_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_audio,
    )


def create_numerals_dataloader(
    manifest_path: str | Path,
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    dataset_dir: str | Path | None = None,
) -> DataLoader[dict[str, Any]]:
    """Create the unlabeled waveform loader used for Phase 1 numerals."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if num_workers < 0:
        raise ValueError("num_workers must not be negative")
    dataset = ParquetNumeralsDataset(manifest_path, dataset_dir=dataset_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_numerals_audio,
    )
