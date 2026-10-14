from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from pi5_vcm.dataset_prepare import (
    INTENT_NAMES,
    SLOT_INTENTS,
    SourceRecord,
    _include_synthetic_negatives,
    _intent_and_slots,
    _read_parquet_records,
    _read_slot_values,
    grouped_stratified_split,
    prepare_dataset,
)
from pi5_vcm.dataset import ParquetAudioDataset, ParquetNumeralsDataset


def source_record(
    clip_id: str, speaker_id: str, intent_id: int
) -> SourceRecord:
    return SourceRecord(
        clip_id=clip_id,
        parquet_path=Path("train.parquet"),
        row_group=0,
        row_index=0,
        intent_id=intent_id,
        slot_targets=(-1,) * 6,
        slot_mask=(0,) * 6,
        speaker_id=speaker_id,
        duration_s=1.0,
        row_id=clip_id,
    )


class GroupedSplitTests(unittest.TestCase):
    def test_split_is_reproducible_speaker_disjoint_and_class_covered(self) -> None:
        records = [
            source_record(f"{speaker}-{class_id}", speaker, class_id)
            for speaker in ("speaker-a", "speaker-b", "speaker-c", "speaker-d")
            for class_id in (0, 1)
        ]

        train_a, validation_a, speakers_a = grouped_stratified_split(
            records, val_fraction=0.25, seed=42
        )
        train_b, validation_b, speakers_b = grouped_stratified_split(
            records, val_fraction=0.25, seed=42
        )

        self.assertEqual(speakers_a, speakers_b)
        self.assertEqual(validation_a, validation_b)
        self.assertEqual(len(validation_a), 2)
        self.assertEqual({row.speaker_id for row in validation_a}, set(speakers_a))
        self.assertFalse(
            {row.speaker_id for row in train_a}
            & {row.speaker_id for row in validation_a}
        )
        self.assertEqual({row.intent_id for row in validation_a}, {0, 1})


class SyntheticNegativeProvenanceTests(unittest.TestCase):
    def test_only_resolved_train_only_provenance_is_included(self) -> None:
        train_source = source_record("audio/train-source.wav", "speaker-train", 0)
        val_source = source_record("audio/val-source.wav", "speaker-val", 0)
        safe_negative = source_record("synthetic_negative:safe.wav", "", 19)
        val_negative = source_record("synthetic_negative:val.wav", "", 19)
        unknown_negative = source_record("synthetic_negative:unknown.wav", "", 19)

        included, counts = _include_synthetic_negatives(
            [train_source],
            [val_source],
            [safe_negative, val_negative, unknown_negative],
            {
                safe_negative.clip_id: ["train/audio/train-source.wav"],
                val_negative.clip_id: ["train/audio/val-source.wav"],
                unknown_negative.clip_id: ["train/audio/missing.wav"],
            },
        )

        self.assertEqual(included, [safe_negative])
        self.assertEqual(counts["included"], 1)
        self.assertEqual(counts["excluded"], 2)
        self.assertEqual(counts["validation_source"], 1)
        self.assertEqual(counts["unresolved_provenance"], 1)


def wav_fixture() -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray([0.1, -0.1], dtype=np.float32), 16_000, format="WAV")
    return buffer.getvalue()


def long_wav_fixture() -> bytes:
    buffer = io.BytesIO()
    sf.write(
        buffer,
        np.zeros(8_001, dtype=np.float32),
        16_000,
        format="WAV",
    )
    return buffer.getvalue()


def write_dataset_fixture(root: Path) -> list[dict[str, object]]:
    (root / "data").mkdir(parents=True)
    (root / "synthetic_negatives").mkdir()
    intents = list(INTENT_NAMES)
    slot_values = {
        intent: [f"{intent.lower()}-value-{index}" for index in range(3)]
        for intent in SLOT_INTENTS
    }

    rows: list[dict[str, object]] = []
    for speaker in ("speaker-a", "speaker-b", "speaker-c", "speaker-d"):
        for intent in intents:
            is_oos = intent == "OUT_OF_SCOPE"
            slot_value = slot_values.get(intent, [""])[0]
            rows.append(
                {
                    "audio": {"bytes": wav_fixture(), "path": "fixture.wav"},
                    "file": f"audio/{speaker}/{intent.lower()}.wav",
                    "command": intent,
                    "slot_value": (
                        "unsupported-slot"
                        if speaker == "speaker-a" and intent == "TIMER"
                        else "" if is_oos else slot_value
                    ),
                    "out_of_scope": int(is_oos),
                    "speaker_id": speaker,
                    "duration_s": 1.0,
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), root / "data/train-00000.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "audio": {"bytes": long_wav_fixture(), "path": "numeral.wav"},
                    "file": "audio/numerals/numeral.wav",
                    "duration_s": 0.5000625,
                }
            ]
        ),
        root / "data/numerals-00000.parquet",
    )

    return rows


class PrepareDatasetTests(unittest.TestCase):
    def test_prepare_writes_hashed_manifests_and_filters_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "snapshot"
            root.mkdir()
            rows = write_dataset_fixture(root)
            validation_speakers = grouped_stratified_split(
                _read_parquet_records(
                    [root / "data/train-00000.parquet"],
                    slot_values={
                        intent: {
                            f"{intent.lower()}-value-{index}": index
                            for index in range(3)
                        }
                        for intent in SLOT_INTENTS
                    },
                ),
                val_fraction=0.25,
                seed=42,
            )[2]
            source_by_speaker = {
                speaker: next(row["file"] for row in rows if speaker in str(row["file"]))
                for speaker in ("speaker-a", "speaker-b", "speaker-c", "speaker-d")
            }
            train_speaker = next(
                speaker
                for speaker in source_by_speaker
                if speaker not in validation_speakers
            )
            validation_speaker = validation_speakers[0]
            negative_rows = [
                {
                    "audio": {"bytes": wav_fixture(), "path": "negative.wav"},
                    "file": "negative-safe.wav",
                    "command": "OUT_OF_SCOPE",
                    "slot_value": "",
                    "out_of_scope": 1,
                    "speaker_id": None,
                    "duration_s": 1.0,
                    "source_files": f"train/{source_by_speaker[train_speaker]}",
                },
                {
                    "audio": {"bytes": wav_fixture(), "path": "negative.wav"},
                    "file": "negative-validation.wav",
                    "command": "OUT_OF_SCOPE",
                    "slot_value": "",
                    "out_of_scope": 1,
                    "speaker_id": None,
                    "duration_s": 1.0,
                    "source_files": f"train/{source_by_speaker[validation_speaker]}",
                },
                {
                    "audio": {"bytes": wav_fixture(), "path": "negative.wav"},
                    "file": "negative-unknown.wav",
                    "command": "OUT_OF_SCOPE",
                    "slot_value": "",
                    "out_of_scope": 1,
                    "speaker_id": None,
                    "duration_s": 1.0,
                    "source_files": "train/audio/unknown.wav",
                },
            ]
            pq.write_table(
                pa.Table.from_pylist(negative_rows),
                root / "synthetic_negatives/train-00000.parquet",
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            **negative_rows[0],
                            "file": "tune-oos.wav",
                            "source_files": "train/audio/speaker-a/play_music.wav",
                        }
                    ]
                ),
                root / "synthetic_negatives/tune_oos-00000.parquet",
            )
            variations_path = root / "variations.csv"
            with variations_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["label", "variation", "value", "phrase"])
                for intent in SLOT_INTENTS:
                    for index in range(3):
                        writer.writerow(
                            [
                                intent,
                                index + 1,
                                f"{intent.lower()}-value-{index}",
                                "fixture",
                            ]
                        )

            output_dir = Path(temporary_directory) / "prepared"
            result = prepare_dataset(root, output_dir, val_fraction=0.25, seed=42)

            train_manifest = [
                json.loads(line)
                for line in (output_dir / "train.jsonl").read_text().splitlines()
            ]
            validation_manifest = [
                json.loads(line)
                for line in (output_dir / "validation.jsonl").read_text().splitlines()
            ]
            tune_manifest = [
                json.loads(line)
                for line in (output_dir / "tune_oos.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(validation_manifest), len(INTENT_NAMES))
            self.assertEqual(
                result["synthetic_negatives"]["included_in_train"], 1
            )
            self.assertEqual(
                result["synthetic_negatives"]["excluded_from_train"], 2
            )
            self.assertEqual(
                sum(row["is_synthetic_negative"] for row in train_manifest), 1
            )
            self.assertTrue(result["split_manifest_sha256"])
            self.assertEqual(
                len(result["split"]["validation_speakers"]), 1
            )
            second_output = Path(temporary_directory) / "prepared-again"
            second_result = prepare_dataset(
                root, second_output, val_fraction=0.25, seed=42
            )
            self.assertEqual(
                result["split_manifest_sha256"],
                second_result["split_manifest_sha256"],
            )
            self.assertEqual(
                (output_dir / "train.jsonl").read_bytes(),
                (second_output / "train.jsonl").read_bytes(),
            )
            self.assertEqual(len(tune_manifest), 1)
            self.assertEqual(tune_manifest[0]["intent_id"], 19)
            self.assertNotIn(
                tune_manifest[0]["clip_id"],
                {row["clip_id"] for row in train_manifest + validation_manifest},
            )
            slot_row = next(row for row in train_manifest if row["intent_id"] == 13)
            if slot_row["speaker_id"] == "speaker-a":
                self.assertEqual(slot_row["slot_targets"], [-1] * 6)
                self.assertEqual(slot_row["slot_mask"], [0] * 6)
            else:
                self.assertEqual(slot_row["slot_targets"][0], 0)
                self.assertEqual(slot_row["slot_mask"], [1, 0, 0, 0, 0, 0])
            first_train_sample = ParquetAudioDataset(
                output_dir / "train.jsonl", dataset_dir=root
            )[0]
            self.assertEqual(first_train_sample["sample_rate"], 16_000)
            self.assertEqual(first_train_sample["audio"].shape, (2,))
            numeral_sample = ParquetNumeralsDataset(
                output_dir / "numerals.jsonl", dataset_dir=root
            )[0]
            self.assertGreater(numeral_sample["duration_s"], 0.5)
            self.assertNotIn("intent_target", numeral_sample)

    def test_unknown_slot_value_disables_all_slot_heads(self) -> None:
        slot_values = {
            intent: {f"{intent.lower()}-{index}": index for index in range(3)}
            for intent in SLOT_INTENTS
        }
        intent_id, targets, masks = _intent_and_slots(
            {"command": "TIMER", "slot_value": "not in schema"},
            slot_values,
        )

        self.assertEqual(intent_id, INTENT_NAMES.index("TIMER"))
        self.assertEqual(targets, (-1,) * 6)
        self.assertEqual(masks, (0,) * 6)

    def test_numerals_loader_rejects_short_stereo_and_nonfinite_audio(self) -> None:
        cases = (
            (np.zeros(8_000, dtype=np.float32), 16_000, "longer than 0.5"),
            (np.zeros((8_001, 2), dtype=np.float32), 16_000, "must be mono"),
            (np.full(8_001, np.nan, dtype=np.float32), 16_000, "NaN or infinity"),
        )
        for samples, sample_rate, expected_message in cases:
            with self.subTest(expected_message=expected_message):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    buffer = io.BytesIO()
                    sf.write(
                        buffer,
                        samples,
                        sample_rate,
                        format="WAV",
                        subtype="FLOAT",
                    )
                    parquet_path = root / "numerals.parquet"
                    pq.write_table(
                        pa.Table.from_pylist(
                            [
                                {
                                    "audio": {
                                        "bytes": buffer.getvalue(),
                                        "path": "numeral.wav",
                                    }
                                }
                            ]
                        ),
                        parquet_path,
                    )
                    manifest_path = root / "numerals.jsonl"
                    manifest_path.write_text(
                        json.dumps(
                            {
                                "clip_id": "numeral",
                                "parquet_path": "numerals.parquet",
                                "row_group": 0,
                                "row_index": 0,
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )

                    with self.assertRaisesRegex(ValueError, expected_message):
                        ParquetNumeralsDataset(manifest_path)[0]


if __name__ == "__main__":
    unittest.main()
