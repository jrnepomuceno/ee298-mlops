from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from pi5_vcm.features import (
    LogMelFrontend,
    compute_mel_statistics,
    load_mel_statistics,
    save_mel_statistics,
)
from pi5_vcm.model import NUM_MEL_BINS


class LogMelFrontendTests(unittest.TestCase):
    def test_generates_aligned_features_and_ignores_padded_waveform_values(self) -> None:
        frontend = LogMelFrontend()
        waveforms = torch.randn(2, 3_200)
        lengths = torch.tensor([3_200, 1_600])
        padded_changed = waveforms.clone()
        padded_changed[1, 1_600:] = 500.0

        features, mask = frontend(waveforms, lengths)
        changed_features, changed_mask = frontend(padded_changed, lengths)

        self.assertEqual(tuple(features.shape), (2, 20, NUM_MEL_BINS))
        self.assertEqual(mask.tolist(), [[True] * 20, [True] * 10 + [False] * 10])
        self.assertTrue(torch.isfinite(features).all().item())
        torch.testing.assert_close(mask, changed_mask)
        torch.testing.assert_close(features[1, :10], changed_features[1, :10])
        self.assertEqual(features[1, 10:].abs().sum().item(), 0.0)

    def test_statistics_fit_only_valid_frames_and_round_trip(self) -> None:
        frontend = LogMelFrontend()
        batches = [
            {
                "audio": torch.randn(2, 3_200),
                "audio_lengths": torch.tensor([3_200, 1_600]),
            }
        ]
        mean, std = compute_mel_statistics(
            frontend, batches, device=torch.device("cpu")
        )

        self.assertEqual(tuple(mean.shape), (NUM_MEL_BINS,))
        self.assertEqual(tuple(std.shape), (NUM_MEL_BINS,))
        self.assertTrue(torch.isfinite(mean).all().item())
        self.assertTrue(torch.all(std > 0).item())
        with tempfile.TemporaryDirectory() as temporary_directory:
            statistics_path = Path(temporary_directory) / "mel_stats.npz"
            save_mel_statistics(statistics_path, mean, std)
            restored_mean, restored_std = load_mel_statistics(statistics_path)
        torch.testing.assert_close(restored_mean, mean)
        torch.testing.assert_close(restored_std, std)


if __name__ == "__main__":
    unittest.main()
