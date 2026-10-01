"""Tests for the v6 intent-plus-bounded-slots model definition."""
from __future__ import annotations

import unittest

import torch

from new_training.intent_v6_model import IntentBoundedSlots


class IntentBoundedSlotsTests(unittest.TestCase):
    def test_output_head_shapes_and_checkpoint_key_layout(self):
        config = {
            "n_mels": 80,
            "num_intents": 18,
            "conv_channels": 64,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.3,
        }
        slot_sizes = (61, 60, 12, 4, 2, 25, 101, 6)
        model = IntentBoundedSlots(config, slot_sizes).eval()
        mels = torch.zeros((2, 80, 80), dtype=torch.float32)
        lengths = torch.tensor([80, 64], dtype=torch.long)

        with torch.no_grad():
            outputs = model(mels, lengths)

        self.assertEqual([tuple(output.shape) for output in outputs], [
            (2, 18), (2, 61), (2, 60), (2, 12), (2, 4),
            (2, 2), (2, 25), (2, 101), (2, 6),
        ])
        keys = model.state_dict()
        self.assertIn("encoder.gru_in.weight", keys)
        self.assertIn("attention.weight", keys)
        self.assertIn("heads.7.weight", keys)


if __name__ == "__main__":
    unittest.main()