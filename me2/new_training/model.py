"""Intent-only model for the capped package training baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class IntentModel(nn.Module):
    def __init__(self, n_mels: int = 80, num_intents: int = 18,
                 conv_channels: int = 64, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.n_mels = n_mels
        self.num_intents = num_intents
        self.conv_channels = conv_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )
        self.gru_in = nn.Linear(conv_channels * 2 * n_mels, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.intent_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_intents),
        )

    def forward(self, mels: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        features = self.conv(mels.unsqueeze(1))
        batch, channels, steps, bins = features.shape
        features = features.permute(0, 2, 1, 3).reshape(batch, steps, channels * bins)
        sequence, _ = self.gru(self.gru_in(features))
        if lengths is None:
            pooled = sequence.mean(dim=1)
        else:
            reduced_lengths = torch.clamp(lengths // 4, min=1)
            mask = torch.arange(steps, device=mels.device).unsqueeze(0) < reduced_lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).to(sequence.dtype)
            pooled = (sequence * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.intent_head(pooled)
