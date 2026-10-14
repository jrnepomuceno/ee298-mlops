from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from pi5_vcm.model import (
    NUM_INTENTS,
    NUM_SLOT_CLASSES,
    NUM_SLOT_HEADS,
    SLOT_INTENT_IDS,
    ModelOutput,
)


def inverse_frequency_class_weights(class_counts: Sequence[int]) -> Tensor:
    """Return balanced inverse-frequency weights for all intent classes."""
    if len(class_counts) != NUM_INTENTS:
        raise ValueError(f"class_counts must contain {NUM_INTENTS} entries")
    if any(count < 0 for count in class_counts):
        raise ValueError("class counts must not be negative")
    total = sum(class_counts)
    if total == 0 or any(count == 0 for count in class_counts):
        raise ValueError("every intent class must have at least one training example")
    return torch.tensor(
        [total / (NUM_INTENTS * count) for count in class_counts],
        dtype=torch.float32,
    )


def multitask_loss(
    outputs: ModelOutput,
    intent_targets: Tensor,
    slot_targets: Tensor,
    slot_mask: Tensor,
    *,
    intent_class_weights: Tensor | None = None,
    slot_loss_weights: Sequence[float] = (1.0,) * NUM_SLOT_HEADS,
) -> dict[str, Tensor]:
    """Compute class-balanced intent CE and per-intent masked slot CE."""
    intent_logits = outputs["intent_logits"]
    slot_logits = outputs["slot_logits"]
    batch_size = intent_logits.shape[0]
    if intent_logits.shape != (batch_size, NUM_INTENTS):
        raise ValueError(f"intent_logits must have shape (batch, {NUM_INTENTS})")
    if slot_logits.shape != (batch_size, NUM_SLOT_HEADS, NUM_SLOT_CLASSES):
        raise ValueError(
            "slot_logits must have shape "
            f"(batch, {NUM_SLOT_HEADS}, {NUM_SLOT_CLASSES})"
        )
    if intent_targets.shape != (batch_size,):
        raise ValueError("intent_targets must have shape (batch,)")
    if slot_targets.shape != (batch_size, NUM_SLOT_HEADS):
        raise ValueError(
            f"slot_targets must have shape (batch, {NUM_SLOT_HEADS})"
        )
    if slot_mask.shape != (batch_size, NUM_SLOT_HEADS):
        raise ValueError(f"slot_mask must have shape (batch, {NUM_SLOT_HEADS})")
    if len(slot_loss_weights) != NUM_SLOT_HEADS:
        raise ValueError(f"slot_loss_weights must contain {NUM_SLOT_HEADS} entries")
    if torch.any((intent_targets < 0) | (intent_targets >= NUM_INTENTS)).item():
        raise ValueError(f"intent_targets must be in [0, {NUM_INTENTS - 1}]")

    intent_loss = F.cross_entropy(
        intent_logits, intent_targets, weight=intent_class_weights
    )
    boolean_slot_mask = slot_mask.to(device=intent_targets.device, dtype=torch.bool)
    slot_losses: list[Tensor] = []
    for index, intent_id in enumerate(SLOT_INTENT_IDS):
        slot_target = slot_targets[:, index]
        active = boolean_slot_mask[:, index] & (intent_targets == intent_id)
        if torch.any(active & ((slot_target < 0) | (slot_target >= NUM_SLOT_CLASSES))).item():
            raise ValueError(
                f"active slot targets for head {index} must be in "
                f"[0, {NUM_SLOT_CLASSES - 1}]"
            )
        if active.any().item():
            slot_loss = F.cross_entropy(slot_logits[active, index], slot_target[active])
        else:
            slot_loss = slot_logits[:, index, :].sum() * 0.0
        slot_losses.append(slot_loss)

    slot_loss_total = sum(
        weight * loss for weight, loss in zip(slot_loss_weights, slot_losses, strict=True)
    )
    return {
        "total": intent_loss + slot_loss_total,
        "intent": intent_loss,
        "slots": slot_loss_total,
    }
