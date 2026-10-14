from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from pi5_vcm.dataset import NUM_INTENTS, NUM_SLOT_HEADS
from pi5_vcm.dataset_source import DATASET_REVISION

INTENT_NAMES = (
    "PLAY_MUSIC",
    "WEATHER",
    "TIME",
    "LIGHT_ON",
    "LIGHT_OFF",
    "PAUSE",
    "STOP",
    "NEXT",
    "VOLUME_UP",
    "VOLUME_DOWN",
    "CALL",
    "MESSAGE",
    "LIST_REMINDERS",
    "TIMER",
    "ALARM",
    "TEMPERATURE",
    "BRIGHTNESS",
    "COLOR",
    "CREATE_REMINDER",
    "OUT_OF_SCOPE",
)
SLOT_INTENTS = (
    "TIMER",
    "ALARM",
    "TEMPERATURE",
    "BRIGHTNESS",
    "COLOR",
    "CREATE_REMINDER",
)
SPLIT_SEED = 42
VALIDATION_FRACTION = 0.10
BASE_SOURCE_COLUMNS = [
    "file",
    "command",
    "slot_value",
    "out_of_scope",
    "speaker_id",
    "duration_s",
]


@dataclass(frozen=True)
class SourceRecord:
    clip_id: str
    parquet_path: Path
    row_group: int
    row_index: int
    intent_id: int
    slot_targets: tuple[int, ...]
    slot_mask: tuple[int, ...]
    speaker_id: str | None
    duration_s: float | None
    row_id: str
    is_synthetic_negative: bool = False

    def as_manifest_entry(self, *, dataset_root: Path) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "parquet_path": self.parquet_path.relative_to(dataset_root).as_posix(),
            "row_group": self.row_group,
            "row_index": self.row_index,
            "intent_id": self.intent_id,
            "slot_targets": list(self.slot_targets),
            "slot_mask": list(self.slot_mask),
            "speaker_id": self.speaker_id,
            "duration_s": self.duration_s,
            "row_id": self.row_id,
            "is_synthetic_negative": self.is_synthetic_negative,
        }


def _normalize_source_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    normalized = re.sub(r"^(\./)+", "", normalized)
    normalized = os.path.normpath(normalized).replace("\\", "/")
    if normalized.startswith("train/"):
        normalized = normalized[len("train/") :]
    if normalized.startswith("audio/"):
        normalized = normalized[len("audio/") :]
    return normalized


def _read_slot_values(variations_path: Path) -> dict[str, dict[str, int]]:
    values_by_intent: dict[str, list[str]] = defaultdict(list)
    with variations_path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            intent = row["label"].strip()
            value = row.get("value", "").strip()
            if intent in SLOT_INTENTS and value:
                key = value.casefold()
                if key not in values_by_intent[intent]:
                    values_by_intent[intent].append(key)

    mappings: dict[str, dict[str, int]] = {}
    for intent in SLOT_INTENTS:
        values = values_by_intent[intent]
        if len(values) != 3:
            raise ValueError(
                f"variations.csv must define exactly three slot values for "
                f"{intent}; found {len(values)}"
            )
        mappings[intent] = {value: index for index, value in enumerate(values)}
    return mappings


def _intent_and_slots(
    row: dict[str, Any], slot_values: dict[str, dict[str, int]]
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    command = str(row.get("command") or "").strip().upper()
    is_oos = bool(row.get("out_of_scope")) or command == "OUT_OF_SCOPE"
    if is_oos:
        return NUM_INTENTS - 1, (-1,) * NUM_SLOT_HEADS, (0,) * NUM_SLOT_HEADS
    if command not in INTENT_NAMES[:-1]:
        raise ValueError(f"Unknown in-scope command label: {command!r}")

    intent_id = INTENT_NAMES.index(command)
    targets = [-1] * NUM_SLOT_HEADS
    masks = [0] * NUM_SLOT_HEADS
    if command in SLOT_INTENTS:
        slot_index = SLOT_INTENTS.index(command)
        slot_value = str(row.get("slot_value") or "").strip().casefold()
        target = slot_values[command].get(slot_value)
        if target is not None:
            targets[slot_index] = target
            masks[slot_index] = 1
    return intent_id, tuple(targets), tuple(masks)


def _read_parquet_records(
    parquet_paths: list[Path],
    *,
    slot_values: dict[str, dict[str, int]],
    is_synthetic_negative: bool = False,
) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    seen_ids: set[str] = set()
    for parquet_path in sorted(parquet_paths):
        parquet_file = pq.ParquetFile(parquet_path)
        required_columns = BASE_SOURCE_COLUMNS.copy()
        if is_synthetic_negative:
            required_columns.append("source_files")
        missing_columns = set(required_columns).difference(
            parquet_file.schema_arrow.names
        )
        if missing_columns:
            raise ValueError(
                f"{parquet_path} is missing required columns: "
                f"{', '.join(sorted(missing_columns))}"
            )
        for row_group_index in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(
                row_group_index, columns=BASE_SOURCE_COLUMNS
            )
            for row_index, row in enumerate(table.to_pylist()):
                row_id = str(row.get("file") or "").strip()
                if not row_id:
                    raise ValueError(
                        f"{parquet_path}, row group {row_group_index}, "
                        f"row {row_index}: missing file identifier"
                    )
                clip_id = (
                    f"synthetic_negative:{row_id}" if is_synthetic_negative else row_id
                )
                if clip_id in seen_ids:
                    raise ValueError(f"Duplicate clip identifier: {clip_id}")
                seen_ids.add(clip_id)
                intent_id, slot_targets, slot_mask = _intent_and_slots(row, slot_values)
                speaker = row.get("speaker_id")
                duration = row.get("duration_s")
                records.append(
                    SourceRecord(
                        clip_id=clip_id,
                        parquet_path=parquet_path,
                        row_group=row_group_index,
                        row_index=row_index,
                        intent_id=intent_id,
                        slot_targets=slot_targets,
                        slot_mask=slot_mask,
                        speaker_id=str(speaker) if speaker not in (None, "") else None,
                        duration_s=float(duration) if duration is not None else None,
                        row_id=row_id,
                        is_synthetic_negative=is_synthetic_negative,
                    )
                )
    if not records:
        raise ValueError("No Parquet records were found")
    return records


def grouped_stratified_split(
    records: list[SourceRecord],
    *,
    val_fraction: float = VALIDATION_FRACTION,
    seed: int = SPLIT_SEED,
) -> tuple[list[SourceRecord], list[SourceRecord], list[str]]:
    """Split original train rows by speaker using the plan's greedy objective."""
    if not records:
        raise ValueError("Cannot split an empty training set")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    rows_by_speaker: dict[str, list[SourceRecord]] = defaultdict(list)
    source_counts: Counter[int] = Counter()
    for record in records:
        if record.speaker_id is None:
            raise ValueError(f"{record.clip_id}: original train row has no speaker_id")
        rows_by_speaker[record.speaker_id].append(record)
        source_counts[record.intent_id] += 1

    speakers = sorted(rows_by_speaker)
    tie_order = speakers.copy()
    random.Random(seed).shuffle(tie_order)
    tie_rank = {speaker: rank for rank, speaker in enumerate(tie_order)}
    speaker_counts = {
        speaker: Counter(record.intent_id for record in speaker_rows)
        for speaker, speaker_rows in rows_by_speaker.items()
    }
    target_size = math.ceil(len(records) * val_fraction)
    validation_speakers: list[str] = []
    validation_counts: Counter[int] = Counter()
    validation_size = 0
    classes_present = set(source_counts)

    while validation_size < target_size or classes_present.difference(validation_counts):
        candidates = [
            speaker for speaker in speakers if speaker not in validation_speakers
        ]
        if not candidates:
            missing = sorted(classes_present.difference(validation_counts))
            raise ValueError(
                "Unable to satisfy grouped split requirements; "
                f"missing validation classes: {missing}"
            )

        def candidate_score(speaker: str) -> tuple[int, float, float, int]:
            new_size = validation_size + len(rows_by_speaker[speaker])
            new_counts = validation_counts + speaker_counts[speaker]
            missing_count = len(classes_present.difference(new_counts))
            size_error = abs(new_size - target_size) / target_size
            class_error = sum(
                abs(
                    new_counts[class_id] / new_size
                    - source_counts[class_id] / len(records)
                )
                for class_id in classes_present
            )
            return missing_count, size_error, class_error, tie_rank[speaker]

        selected = min(candidates, key=candidate_score)
        validation_speakers.append(selected)
        selected_rows = rows_by_speaker[selected]
        validation_size += len(selected_rows)
        validation_counts.update(speaker_counts[selected])

    validation_speaker_set = set(validation_speakers)
    train_records = [
        record for record in records if record.speaker_id not in validation_speaker_set
    ]
    validation_records = [
        record for record in records if record.speaker_id in validation_speaker_set
    ]
    return train_records, validation_records, validation_speakers


def _source_files(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    stripped = value.strip()
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        return [item.strip() for item in parsed if isinstance(item, str) and item.strip()]
    return [item.strip() for item in stripped.split(";") if item.strip()]


def _include_synthetic_negatives(
    train_records: list[SourceRecord],
    validation_records: list[SourceRecord],
    negative_records: list[SourceRecord],
    provenance: dict[str, list[str]],
) -> tuple[list[SourceRecord], dict[str, int]]:
    source_index: dict[str, set[SourceRecord]] = defaultdict(set)
    for record in train_records + validation_records:
        source_index[_normalize_source_path(record.row_id)].add(record)

    validation_speakers = {record.speaker_id for record in validation_records}
    included: list[SourceRecord] = []
    reasons: Counter[str] = Counter()
    for negative in negative_records:
        raw_sources = provenance.get(negative.clip_id, [])
        if not raw_sources:
            reasons["missing_provenance"] += 1
            continue

        resolved: list[SourceRecord] = []
        unresolved = False
        for source in raw_sources:
            matches = source_index.get(_normalize_source_path(source), set())
            if len(matches) != 1:
                unresolved = True
                break
            resolved.append(next(iter(matches)))
        if unresolved:
            reasons["unresolved_provenance"] += 1
            continue
        source_speakers = {record.speaker_id for record in resolved}
        if source_speakers.intersection(validation_speakers):
            reasons["validation_source"] += 1
            continue
        if None in source_speakers:
            reasons["missing_source_speaker"] += 1
            continue
        included.append(negative)
    reasons["included"] = len(included)
    reasons["excluded"] = len(negative_records) - len(included)
    return included, dict(sorted(reasons.items()))


def _read_negative_records(
    parquet_paths: list[Path],
    *,
    slot_values: dict[str, dict[str, int]],
) -> tuple[list[SourceRecord], dict[str, list[str]]]:
    records = _read_parquet_records(
        parquet_paths,
        slot_values=slot_values,
        is_synthetic_negative=True,
    )
    provenance: dict[str, list[str]] = {}
    for parquet_path in sorted(parquet_paths):
        parquet_file = pq.ParquetFile(parquet_path)
        for row_group_index in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(
                row_group_index, columns=["file", "source_files"]
            )
            for row in table.to_pylist():
                clip_id = f"synthetic_negative:{str(row.get('file') or '').strip()}"
                provenance[clip_id] = _source_files(row.get("source_files"))
    return records, provenance


def _numerals_manifest(parquet_paths: list[Path], dataset_root: Path) -> bytes:
    entries: list[bytes] = []
    seen_ids: set[str] = set()
    for parquet_path in sorted(parquet_paths):
        parquet_file = pq.ParquetFile(parquet_path)
        if "file" not in parquet_file.schema_arrow.names:
            raise ValueError(f"{parquet_path} is missing required file identifiers")
        for row_group_index in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_index, columns=["file"])
            for row_index, row in enumerate(table.to_pylist()):
                clip_id = str(row.get("file") or "").strip()
                if not clip_id:
                    raise ValueError(
                        f"{parquet_path}, row group {row_group_index}, "
                        f"row {row_index}: missing file identifier"
                    )
                if clip_id in seen_ids:
                    raise ValueError(f"Duplicate numeral clip identifier: {clip_id}")
                seen_ids.add(clip_id)
                entry = {
                    "clip_id": clip_id,
                    "parquet_path": parquet_path.relative_to(dataset_root).as_posix(),
                    "row_group": row_group_index,
                    "row_index": row_index,
                }
                entries.append(_canonical_json(entry) + b"\n")
    if not entries:
        raise ValueError("No numeral Parquet records were found")
    return b"".join(entries)


def prepare_dataset(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    val_fraction: float = VALIDATION_FRACTION,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    dataset_root = Path(dataset_dir).resolve()
    output_root = Path(output_dir).resolve()
    train_paths = sorted((dataset_root / "data").glob("train-*.parquet"))
    numerals_paths = sorted((dataset_root / "data").glob("numerals-*.parquet"))
    negative_paths = sorted(
        (dataset_root / "synthetic_negatives").glob("train-*.parquet")
    )
    tune_paths = sorted(
        (dataset_root / "synthetic_negatives").glob("tune_oos-*.parquet")
    )
    variations_path = dataset_root / "variations.csv"
    if not train_paths:
        raise FileNotFoundError(f"No original training Parquet files under {dataset_root}")
    if not numerals_paths:
        raise FileNotFoundError(f"No numerals Parquet files under {dataset_root}")
    if not negative_paths:
        raise FileNotFoundError(
            f"No synthetic-negative training Parquet files under {dataset_root}"
        )
    if not tune_paths:
        raise FileNotFoundError(
            f"No synthetic tune_oos Parquet files under {dataset_root}"
        )
    if not variations_path.is_file():
        raise FileNotFoundError(f"Missing slot schema file: {variations_path}")

    slot_values = _read_slot_values(variations_path)
    original_records = _read_parquet_records(train_paths, slot_values=slot_values)
    train_records, validation_records, validation_speakers = grouped_stratified_split(
        original_records,
        val_fraction=val_fraction,
        seed=seed,
    )
    negative_records, negative_metadata = _read_negative_records(
        negative_paths, slot_values=slot_values
    )
    tune_records, _ = _read_negative_records(tune_paths, slot_values=slot_values)
    if any(record.intent_id != NUM_INTENTS - 1 for record in tune_records):
        raise ValueError("Every synthetic tune_oos row must have OUT_OF_SCOPE label")
    included_negatives, negative_counts = _include_synthetic_negatives(
        train_records, validation_records, negative_records, negative_metadata
    )
    train_records.extend(included_negatives)
    output_root.mkdir(parents=True, exist_ok=True)

    train_jsonl = _jsonl(train_records, dataset_root)
    validation_jsonl = _jsonl(validation_records, dataset_root)
    tune_jsonl = _jsonl(tune_records, dataset_root)
    numerals_jsonl = _numerals_manifest(numerals_paths, dataset_root)
    _write_atomic(output_root / "train.jsonl", train_jsonl)
    _write_atomic(output_root / "validation.jsonl", validation_jsonl)
    _write_atomic(output_root / "tune_oos.jsonl", tune_jsonl)
    _write_atomic(output_root / "numerals.jsonl", numerals_jsonl)

    original_train_counts = Counter(
        record.intent_id
        for record in train_records
        if not record.is_synthetic_negative
    )
    validation_counts = Counter(record.intent_id for record in validation_records)
    payload: dict[str, Any] = {
        "dataset": {
            "repo_id": "airimonda/ai231-me2-voice-commands",
            "revision": DATASET_REVISION,
        },
        "split": {
            "algorithm": "speaker_grouped_greedy",
            "seed": seed,
            "validation_fraction_target": val_fraction,
            "original_train_rows": len(original_records),
            "validation_target_rows": math.ceil(
                len(original_records) * val_fraction
            ),
            "validation_rows": len(validation_records),
            "validation_fraction_achieved": (
                len(validation_records) / len(original_records)
            ),
            "validation_speakers": sorted(validation_speakers),
            "train_original_speakers": sorted(
                {
                    record.speaker_id
                    for record in train_records
                    if not record.is_synthetic_negative
                }
            ),
            "validation_row_ids": [record.row_id for record in validation_records],
            "train_original_row_ids": [
                record.row_id
                for record in train_records
                if not record.is_synthetic_negative
            ],
            "train_original_class_counts": {
                INTENT_NAMES[class_id]: original_train_counts[class_id]
                for class_id in range(NUM_INTENTS)
            },
            "source_class_counts": {
                INTENT_NAMES[class_id]: sum(
                    record.intent_id == class_id for record in original_records
                )
                for class_id in range(NUM_INTENTS)
            },
            "validation_class_counts": {
                INTENT_NAMES[class_id]: validation_counts[class_id]
                for class_id in range(NUM_INTENTS)
            },
        },
        "synthetic_negatives": {
            "source_rows": len(negative_records),
            "included_in_train": negative_counts["included"],
            "excluded_from_train": negative_counts["excluded"],
            "exclusion_reasons": {
                reason: count
                for reason, count in negative_counts.items()
                if reason not in {"included", "excluded"}
            },
        },
        "manifests": {
            "train.jsonl_sha256": hashlib.sha256(train_jsonl).hexdigest(),
            "validation.jsonl_sha256": hashlib.sha256(validation_jsonl).hexdigest(),
            "tune_oos.jsonl_sha256": hashlib.sha256(tune_jsonl).hexdigest(),
            "numerals.jsonl_sha256": hashlib.sha256(numerals_jsonl).hexdigest(),
            "numerals_rows": len(numerals_jsonl.splitlines()),
        },
    }
    payload_bytes = _canonical_json(payload)
    split_hash = hashlib.sha256(payload_bytes).hexdigest()
    payload["split_manifest_sha256"] = split_hash
    _write_atomic(output_root / "split_manifest.json", _canonical_json(payload) + b"\n")
    return payload


def _jsonl(records: list[SourceRecord], dataset_root: Path) -> bytes:
    return b"".join(
        _canonical_json(record.as_manifest_entry(dataset_root=dataset_root)) + b"\n"
        for record in records
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _write_atomic(path: Path, contents: bytes) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_bytes(contents)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reproducible, speaker-disjoint ME2 training manifests."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Pinned Hugging Face dataset snapshot directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for train.jsonl, validation.jsonl, and split_manifest.json.",
    )
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--validation-fraction", type=float, default=VALIDATION_FRACTION)
    args = parser.parse_args()
    result = prepare_dataset(
        args.dataset_dir,
        args.output_dir,
        val_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(
        f"Prepared {result['split']['original_train_rows']} original rows: "
        f"{result['split']['validation_rows']} validation, "
        f"{result['split']['original_train_rows'] - result['split']['validation_rows']} "
        f"original train, {result['synthetic_negatives']['included_in_train']} "
        f"provenance-safe synthetic negatives."
    )
    print(f"Split manifest SHA-256: {result['split_manifest_sha256']}")


if __name__ == "__main__":
    main()
