from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from pi5_vcm.dataset import ParquetAudioDataset, create_dataloader


def make_wav(samples: list[float]) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(samples, dtype=np.float32), 16_000, format="WAV")
    return buffer.getvalue()


class ParquetAudioDatasetTests(unittest.TestCase):
    def test_reads_audio_and_collates_variable_length_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            parquet_path = root / "audio.parquet"
            manifest_path = root / "train.jsonl"
            table = pa.Table.from_pylist(
                [
                    {"audio": {"bytes": make_wav([0.25, -0.25]), "path": "a.wav"}},
                    {
                        "audio": {
                            "bytes": make_wav([0.5, 0.0, -0.5]),
                            "path": "b.wav",
                        }
                    },
                ]
            )
            pq.write_table(table, parquet_path, row_group_size=1)
            rows = [
                {
                    "clip_id": "clip-a",
                    "parquet_path": "audio.parquet",
                    "row_group": 0,
                    "row_index": 0,
                    "intent_id": 0,
                    "slot_targets": [-1, -1, -1, -1, -1, -1],
                    "slot_mask": [0, 0, 0, 0, 0, 0],
                    "speaker_id": "speaker-a",
                },
                {
                    "clip_id": "clip-b",
                    "parquet_path": "audio.parquet",
                    "row_group": 1,
                    "row_index": 0,
                    "intent_id": 19,
                    "slot_targets": [0, 1, 2, 0, 1, 2],
                    "slot_mask": [1, 1, 1, 1, 1, 1],
                    "speaker_id": "speaker-b",
                },
            ]
            manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            dataset = ParquetAudioDataset(manifest_path)
            first = dataset[0]
            self.assertEqual(first["audio"].shape, (2,))
            self.assertEqual(first["sample_rate"], 16_000)
            self.assertEqual(first["intent_target"].item(), 0)

            batch = next(
                iter(create_dataloader(manifest_path, batch_size=2, num_workers=0))
            )
            self.assertEqual(tuple(batch["audio"].shape), (2, 3))
            self.assertEqual(batch["audio_lengths"].tolist(), [2, 3])
            self.assertEqual(
                batch["audio_mask"].tolist(),
                [[True, True, False], [True, True, True]],
            )
            self.assertEqual(batch["intent_targets"].tolist(), [0, 19])
            self.assertEqual(batch["slot_targets"][0].tolist(), [-1] * 6)
            self.assertEqual(batch["slot_mask"][0].tolist(), [False] * 6)
            self.assertEqual(batch["clip_ids"], ["clip-a", "clip-b"])
            self.assertEqual(batch["audio"][0, 2].item(), 0.0)

    def test_rejects_invalid_slot_mask_for_unsupported_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "invalid.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "clip_id": "bad",
                        "parquet_path": "missing.parquet",
                        "row_group": 0,
                        "row_index": 0,
                        "intent_id": 0,
                        "slot_targets": [-1, 0, 0, 0, 0, 0],
                        "slot_mask": [1, 1, 1, 1, 1, 1],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unsupported slot value"):
                ParquetAudioDataset(manifest_path)


if __name__ == "__main__":
    unittest.main()
