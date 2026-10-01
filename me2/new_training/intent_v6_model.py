"""Model definition for the v6 bounded numeric-slot checkpoint."""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .model import IntentModel


class IntentBoundedSlots(nn.Module):
    """v6 intent encoder with attention-pooled bounded slot heads."""

    def __init__(self, model_config: dict, slot_sizes: Sequence[int]) -> None:
        super().__init__()
        self.encoder = IntentModel(**model_config)
        width = self.encoder.hidden_size * 2
        self.attention = nn.Linear(width, 1)
        self.heads = nn.ModuleList(nn.Linear(width, size) for size in slot_sizes)

    def forward(self, mels: torch.Tensor,
                lengths: torch.Tensor) -> tuple[torch.Tensor, ...]:
        sequence = self.encoder.forward_sequence(mels)
        steps = sequence.shape[1]
        valid = (torch.arange(steps, device=mels.device).unsqueeze(0)
                 < (lengths // 4).clamp_min(1).unsqueeze(1))
        scores = self.attention(sequence).squeeze(-1).masked_fill(~valid, -1e4)
        pooled = (sequence * scores.softmax(1).unsqueeze(-1)).sum(1)
        return (self.encoder.intent_head(pooled),
                *(head(pooled) for head in self.heads))