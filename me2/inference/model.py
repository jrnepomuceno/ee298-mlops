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

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H). Returns (B, H) via mean over time (ignores padding
        implicitly -- padded frames are zero and contribute little)."""
        return x.mean(dim=1)

    def forward(self, mels: torch.Tensor):
        """mels: (B, T, n_mels). Returns (intent_logits, ctc_logits)."""
        x = mels.unsqueeze(1)                     # (B, 1, T, F)
        x = self.conv(x)                          # (B, C, T/4, F)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        x = self.gru_in(x)                        # (B, T/4, hidden_size)
        x, _ = self.gru(x)                        # (B, T/4, 2H)
        intent_logits = self.intent_head(self._pool(x))   # (B, num_intents)
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
    """Rule-based slot parser: (intent, transcript) -> slot dict.

    This is the non-LLM replacement for "the LLM extracts the slots".
    It only ever sees the constrained-vocab transcript, so regexes are
    small and deterministic.
    """
    import re

    slots: dict = {}
    t = transcript.lower()

    m = re.search(r"(\d+)\s*percent", t)
    if m:
        slots["percent"] = int(m.group(1))
    m = re.search(r"(\d+)\s*degrees?", t)
    if m:
        slots["temperature"] = int(m.group(1))
    m = re.search(r"(\d+)\s*(minutes?|seconds?)", t)
    if m:
        slots["duration"] = int(m.group(1))
        slots["duration_unit"] = m.group(2).rstrip("s")
    m = re.search(r"(\d+)\s*(am|pm)", t)
    if m:
        slots["time"] = f"{m.group(1)}:00 {m.group(2)}"
    if intent == "call":
        m = re.search(r"call\s+(\w+)", t)
        if m:
            slots["contact"] = m.group(1)
    if intent == "remind":
        m = re.search(r"remind\s+me\s+(?:to|about)\s+(.+)", t)
        if m:
            slots["note"] = m.group(1).strip()
    if intent == "set_alarm" and "time" not in slots:
        m = re.search(r"(\d+)", t)
        if m:
            slots["time"] = f"{m.group(1)}:00"
    return slots
