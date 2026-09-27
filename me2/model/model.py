"""VCM -- the command model.

One trained model, two heads, no LLM:

    log-mel (T, 80)
        -> 2D conv stack (temporal downsampling x4)
        -> bidirectional GRU (context)
        -> intent head : (B, num_intents)   -- classification
        -> slot head   : (B, T/4, ctc_vocab) -- CTC over a constrained vocab

The constrained CTC vocabulary (digits + clock words + slot words, see
config.py) is what replaces an LLM: decoding gives you the transcript,
and a small rule-based parser turns (intent, transcript) into slots.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config


class VCM(nn.Module):
    def __init__(self,
                 n_mels: int = config.N_MELS,
                 num_intents: int = config.NUM_INTENTS,
                 ctc_vocab_size: int = config.CTC_VOCAB_SIZE,
                 conv_channels: int = 64,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.conv_channels = conv_channels
        self.hidden_size = hidden_size
        self.temporal_downsample = 4  # two stride-2 convs

        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),                     # T // 2, keep all 80 mels
            nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),                     # T // 4, keep all 80 mels
        )
        self.conv_out_dim = conv_channels * 2 * n_mels  # 64*2*80 = 10240

        # Project the 10240-dim conv output down to hidden_size before the
        # GRU: without this the GRU's input projection alone is ~7.9M params
        # (95% of the model) and blows the <= 6 MB int8 budget in background.md.
        self.gru_in = nn.Linear(self.conv_out_dim, hidden_size)
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

        self.slot_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, ctc_vocab_size),
        )

    def _pool(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, H). Returns (B, H) via mean over time.

        When ``lengths`` (B,) true frame counts are given, the mean is
        length-aware: zero-padded frames are masked out instead of
        diluting the pooled vector. Without lengths, falls back to a plain
        mean (pre-padding behavior).
        """
        if lengths is None:
            return x.mean(dim=1)
        t = x.size(1)
        mask = torch.arange(t, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).to(x.dtype)                    # (B, T, 1)
        denom = mask.sum(dim=1).clamp_min(1.0)                   # (B, 1)
        return (x * mask).sum(dim=1) / denom

    def forward(self, mels: torch.Tensor,
                lengths: torch.Tensor | None = None):
        """mels: (B, T, n_mels); lengths: (B,) true mel-frame counts.

        Returns (intent_logits, ctc_logits). When ``lengths`` is given the
        intent pool masks padding and the caller can use it for CTC
        ``input_lengths`` (true length, not padded).
        """
        x = mels.unsqueeze(1)                     # (B, 1, T, F)
        x = self.conv(x)                          # (B, C, T/4, F)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        x = self.gru_in(x)                        # (B, T/4, hidden_size)
        x, _ = self.gru(x)                        # (B, T/4, 2H)
        pooled_lengths = None
        if lengths is not None:
            # conv downsamples time by 4 (two stride-2 pools)
            pooled_lengths = torch.clamp(lengths // 4, min=1)
        intent_logits = self.intent_head(self._pool(x, pooled_lengths))  # (B, num_intents)
        ctc_logits = self.slot_head(x)            # (B, T/4, ctc_vocab)
        return intent_logits, ctc_logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def ctc_greedy_decode(ctc_logits: torch.Tensor,
                      vocab: list[str] | None = None) -> list[str]:
    """Greedy CTC decode (no beam): argmax -> collapse repeats -> drop blanks.

    ctc_logits: (T, V) for one utterance. Returns list of tokens.
    """
    if vocab is None:
        vocab = config.CTC_VOCAB
    ids = ctc_logits.argmax(dim=-1).tolist()
    tokens: list[str] = []
    prev = -1
    for i in ids:
        if i != prev and i != config.CTC_BLANK:
            tokens.append(vocab[i])
        prev = i
    return tokens


def ctc_decode_batch(ctc_logits: torch.Tensor,
                     vocab: list[str] | None = None) -> list[str]:
    """ctc_logits: (B, T, V) -> list of decoded strings."""
    return [" ".join(ctc_greedy_decode(ctc_logits[i], vocab))
            for i in range(ctc_logits.shape[0])]


def parse_slots(intent: str, transcript: str) -> dict:
    """Rule-based slot parser (see ``model.slots.parse_slots``).

    Re-exported here so ``from model import parse_slots`` keeps working; the
    implementation lives in the torch-free ``model.slots`` module so the
    on-device ONNX path can use it without torch.
    """
    from model.slots import parse_slots as _parse_slots
    return _parse_slots(intent, transcript)
