from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from pi5_vcm.model import NUM_SLOT_HEADS
from pi5_vcm.training import (
    macro_f1_score,
    resolve_device,
    resolve_precision,
    train_model,
)


class TrainingTests(unittest.TestCase):
    def test_device_and_precision_resolution(self) -> None:
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))
        self.assertEqual(resolve_precision("auto", torch.device("cpu")), (None, "fp32"))
        self.assertEqual(
            resolve_precision("auto", torch.device("cuda")),
            (torch.float16, "fp16"),
        )
        with patch("pi5_vcm.training.torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), torch.device("cpu"))
            with self.assertRaisesRegex(ValueError, "CUDA is unavailable"):
                resolve_device("cuda:0")
        with self.assertRaisesRegex(ValueError, "supported only on CUDA"):
            resolve_precision("fp16", torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "invalid device"):
            resolve_device("mps")

    def test_macro_f1(self) -> None:
        score = macro_f1_score(torch.tensor([0, 0, 1]), torch.tensor([0, 1, 1]))
        self.assertAlmostEqual(score, 2.0 / 3.0)

    def test_two_stage_training_wires_waveforms_to_checkpoint(self) -> None:
        samples = [
            {
                "audio": torch.randn(3_200),
                "audio_lengths": torch.tensor(3_200),
                "intent_targets": torch.tensor(index),
                "slot_targets": torch.full((NUM_SLOT_HEADS,), -1),
                "slot_mask": torch.zeros(NUM_SLOT_HEADS, dtype=torch.bool),
            }
            for index in (0, 1)
        ]
        loader = DataLoader(samples, batch_size=2)

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = train_model(
                loader,
                loader,
                class_counts=[1] * 20,
                output_dir=temporary_directory,
                stage_a_epochs=1,
                stage_b_epochs=1,
                patience=1,
            )
            checkpoint = torch.load(
                Path(result["checkpoint"]),
                map_location="cpu",
                weights_only=True,
            )
            self.assertTrue(Path(result["mel_statistics"]).is_file())
            self.assertTrue(Path(result["run_report"]).is_file())
            repeated_result = train_model(
                loader,
                loader,
                class_counts=[1] * 20,
                output_dir=Path(temporary_directory) / "repeated",
                stage_a_epochs=1,
                stage_b_epochs=1,
                patience=1,
            )
            repeated_checkpoint = torch.load(
                Path(repeated_result["checkpoint"]),
                map_location="cpu",
                weights_only=True,
            )

        self.assertEqual(result["epochs_completed"], 2)
        self.assertEqual(result["best_stage_b_epoch"], 1)
        self.assertIn("model_state_dict", checkpoint)
        self.assertEqual(tuple(checkpoint["mel_mean"].shape), (80,))
        self.assertEqual(tuple(checkpoint["mel_std"].shape), (80,))
        self.assertEqual(checkpoint["run_metadata"]["seed"], 42)
        self.assertEqual(checkpoint["run_metadata"]["precision"], "fp32")
        self.assertIn("source_sha256", checkpoint["run_metadata"])
        self.assertEqual(
            result["best_validation_macro_f1"],
            repeated_result["best_validation_macro_f1"],
        )
        for parameter_name, parameter in checkpoint["model_state_dict"].items():
            torch.testing.assert_close(
                parameter,
                repeated_checkpoint["model_state_dict"][parameter_name],
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main()
