from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from pi5_vcm.loss import inverse_frequency_class_weights, multitask_loss
from pi5_vcm.model import (
    DS_CNN_BLOCKS,
    EMBEDDING_DIM,
    ME2Model,
    NUM_INTENTS,
    NUM_MEL_BINS,
    NUM_SLOT_CLASSES,
    NUM_SLOT_HEADS,
    count_parameters,
)


class ME2ModelTests(unittest.TestCase):
    def test_output_shapes_and_tiny_parameter_budget(self) -> None:
        model = ME2Model()
        outputs = model(torch.randn(2, 40, NUM_MEL_BINS))

        self.assertEqual(tuple(outputs["intent_logits"].shape), (2, NUM_INTENTS))
        self.assertEqual(
            tuple(outputs["slot_logits"].shape),
            (2, NUM_SLOT_HEADS, NUM_SLOT_CLASSES),
        )
        self.assertEqual(DS_CNN_BLOCKS, 13)
        self.assertEqual(EMBEDDING_DIM, 256)
        self.assertLess(count_parameters(model), 1_000_000)
        self.assertGreater(count_parameters(model), 900_000)

    def test_encoder_is_causal_at_each_frame(self) -> None:
        encoder = ME2Model().encoder.eval()
        features = torch.randn(1, 32, NUM_MEL_BINS)
        changed_future = features.clone()
        changed_future[:, 20:] += 50

        with torch.no_grad():
            original = encoder(features)
            changed = encoder(changed_future)

        torch.testing.assert_close(original[:, :20], changed[:, :20])

    def test_pooling_ignores_batch_padding(self) -> None:
        model = ME2Model().eval()
        features = torch.randn(1, 24, NUM_MEL_BINS)
        padded = F.pad(features, (0, 0, 0, 11), value=7.0)

        with torch.no_grad():
            unpadded_output = model(features)
            padded_output = model(
                padded,
                torch.tensor([[True] * 24 + [False] * 11]),
            )

        torch.testing.assert_close(
            unpadded_output["intent_logits"], padded_output["intent_logits"]
        )
        torch.testing.assert_close(
            unpadded_output["slot_logits"], padded_output["slot_logits"]
        )

    def test_stage_a_freezes_only_encoder_parameters(self) -> None:
        model = ME2Model()
        model.set_encoder_trainable(False)
        self.assertEqual(count_parameters(model, trainable_only=True), 9_766)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.encoder.parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.intent_head.parameters())
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for head in model.slot_heads
                for parameter in head.parameters()
            )
        )

    def test_rejects_examples_without_valid_frames(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one valid frame"):
            ME2Model()(
                torch.randn(1, 4, NUM_MEL_BINS),
                torch.zeros(1, 4, dtype=torch.bool),
            )


class ME2LossTests(unittest.TestCase):
    def test_class_weights_are_inverse_frequency_and_mean_one(self) -> None:
        weights = inverse_frequency_class_weights([1] * NUM_INTENTS)
        torch.testing.assert_close(weights, torch.ones(NUM_INTENTS))
        counts = [2] + [1] * (NUM_INTENTS - 1)
        skewed = inverse_frequency_class_weights(counts)
        self.assertLess(skewed[0], skewed[1])
        weighted_mean = sum(
            count * weight.item() for count, weight in zip(counts, skewed, strict=True)
        ) / sum(counts)
        self.assertAlmostEqual(weighted_mean, 1.0)

    def test_slot_loss_only_uses_matching_intent_and_masked_examples(self) -> None:
        intent_logits = torch.zeros(3, NUM_INTENTS, requires_grad=True)
        slot_logits = torch.zeros(3, NUM_SLOT_HEADS, NUM_SLOT_CLASSES, requires_grad=True)
        with torch.no_grad():
            slot_logits[1, 0] = torch.tensor([0.0, 0.0, 8.0])
        targets = torch.tensor([0, 13, 13])
        slot_targets = torch.full((3, NUM_SLOT_HEADS), -1)
        slot_targets[0, 0] = 2
        slot_targets[1, 0] = 0
        slot_targets[2, 0] = 1
        slot_mask = torch.zeros(3, NUM_SLOT_HEADS, dtype=torch.bool)
        slot_mask[1:, 0] = True
        slot_mask[0, 0] = True

        losses = multitask_loss(
            {"intent_logits": intent_logits, "slot_logits": slot_logits},
            targets,
            slot_targets,
            slot_mask,
        )
        expected_slot_loss = F.cross_entropy(
            slot_logits[[1, 2], 0], torch.tensor([0, 1])
        )

        torch.testing.assert_close(losses["slots"], expected_slot_loss)
        torch.testing.assert_close(
            losses["total"], losses["intent"] + losses["slots"]
        )

    def test_no_active_slots_produces_finite_differentiable_loss(self) -> None:
        intent_logits = torch.randn(2, NUM_INTENTS, requires_grad=True)
        slot_logits = torch.randn(
            2, NUM_SLOT_HEADS, NUM_SLOT_CLASSES, requires_grad=True
        )
        losses = multitask_loss(
            {"intent_logits": intent_logits, "slot_logits": slot_logits},
            torch.tensor([0, 19]),
            torch.full((2, NUM_SLOT_HEADS), -1),
            torch.zeros(2, NUM_SLOT_HEADS, dtype=torch.bool),
        )
        losses["total"].backward()

        self.assertTrue(torch.isfinite(losses["total"]).item())
        self.assertIsNotNone(slot_logits.grad)
        self.assertEqual(slot_logits.grad.abs().sum().item(), 0.0)

    def test_rejects_zero_count_intent_class_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one training example"):
            inverse_frequency_class_weights([1] * (NUM_INTENTS - 1) + [0])


if __name__ == "__main__":
    unittest.main()
