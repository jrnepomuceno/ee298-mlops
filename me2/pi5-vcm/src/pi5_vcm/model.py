from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor, nn
from torch.nn import functional as F

NUM_MEL_BINS = 80
EMBEDDING_DIM = 256
NUM_INTENTS = 20
NUM_SLOT_HEADS = 6
NUM_SLOT_CLASSES = 3
SLOT_NAMES = (
    "TIMER",
    "ALARM",
    "TEMPERATURE",
    "BRIGHTNESS",
    "COLOR",
    "CREATE_REMINDER",
)
SLOT_INTENT_IDS = (13, 14, 15, 16, 17, 18)
DS_CNN_BLOCKS = 13


class ModelOutput(TypedDict):
    intent_logits: Tensor
    slot_logits: Tensor


class CausalConv1d(nn.Module):
    """A stride-one convolution that only reads the current and prior frames."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(F.pad(inputs, (self.left_padding, 0)))


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int) -> None:
        super().__init__()
        self.depthwise = CausalConv1d(
            channels,
            channels,
            kernel_size=5,
            dilation=dilation,
            groups=channels,
        )
        self.depthwise_norm = nn.LayerNorm(channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.pointwise_norm = nn.LayerNorm(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.depthwise(inputs)
        outputs = self.depthwise_norm(outputs.transpose(1, 2)).transpose(1, 2)
        outputs = F.relu(outputs)
        outputs = self.pointwise(outputs)
        outputs = self.pointwise_norm(outputs.transpose(1, 2)).transpose(1, 2)
        return F.relu(inputs + outputs)


class CausalDSCNNEncoder(nn.Module):
    """Map log-mel frames to aligned causal frame features and pooled embeddings."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = CausalConv1d(
            NUM_MEL_BINS, EMBEDDING_DIM, kernel_size=5, bias=True
        )
        self.stem_norm = nn.LayerNorm(EMBEDDING_DIM)
        self.blocks = nn.ModuleList(
            DepthwiseSeparableBlock(
                EMBEDDING_DIM, dilation=(1, 2, 4, 8)[index % 4]
            )
            for index in range(DS_CNN_BLOCKS)
        )

    def forward(self, mel_features: Tensor) -> Tensor:
        if mel_features.ndim != 3 or mel_features.shape[-1] != NUM_MEL_BINS:
            raise ValueError(
                f"mel_features must have shape (batch, frames, {NUM_MEL_BINS})"
            )
        if mel_features.shape[0] == 0 or mel_features.shape[1] == 0:
            raise ValueError("mel_features must contain at least one batch and frame")

        outputs = self.stem(mel_features.transpose(1, 2))
        outputs = self.stem_norm(outputs.transpose(1, 2)).transpose(1, 2)
        outputs = F.relu(outputs)
        for block in self.blocks:
            outputs = block(outputs)
        return outputs.transpose(1, 2)


class ME2Model(nn.Module):
    """Tiny causal encoder with the planned ME2 intent and slot output contract.

    Input log-mel features have shape ``(B, T, 80)``. ``frame_mask`` marks real
    frames in padded batches; the encoder returns ``(B, T, 256)`` frame features
    and pools only valid frames before applying the utterance-level heads.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = CausalDSCNNEncoder()
        self.intent_head = nn.Linear(EMBEDDING_DIM, NUM_INTENTS)
        self.slot_heads = nn.ModuleList(
            nn.Linear(EMBEDDING_DIM, NUM_SLOT_CLASSES)
            for _ in range(NUM_SLOT_HEADS)
        )

    def forward(
        self, mel_features: Tensor, frame_mask: Tensor | None = None
    ) -> ModelOutput:
        frame_features = self.encoder(mel_features)
        if frame_mask is None:
            frame_mask = torch.ones(
                frame_features.shape[:2],
                dtype=torch.bool,
                device=frame_features.device,
            )
        elif frame_mask.shape != frame_features.shape[:2]:
            raise ValueError(
                "frame_mask must have shape (batch, frames) matching mel_features"
            )
        else:
            frame_mask = frame_mask.to(device=frame_features.device, dtype=torch.bool)

        valid_frame_counts = frame_mask.sum(dim=1)
        if torch.any(valid_frame_counts == 0).item():
            raise ValueError("each example must contain at least one valid frame")
        pooled = (
            frame_features * frame_mask.unsqueeze(-1)
        ).sum(dim=1) / valid_frame_counts.unsqueeze(-1)
        return {
            "intent_logits": self.intent_head(pooled),
            "slot_logits": torch.stack(
                [head(pooled) for head in self.slot_heads], dim=1
            ),
        }

    def set_encoder_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze the encoder for supervised training Stage A/B."""
        self.encoder.requires_grad_(trainable)


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Count parameters, optionally counting only those currently trainable."""
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )
