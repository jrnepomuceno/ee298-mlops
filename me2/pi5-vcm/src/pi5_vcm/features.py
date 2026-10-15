from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pi5_vcm.dataset import SAMPLE_RATE
from pi5_vcm.model import NUM_MEL_BINS

FRAME_LENGTH = 400
FRAME_HOP = 160
FFT_SIZE = FRAME_LENGTH
MIN_LOG_ENERGY = 1e-10


def _hz_to_mel(frequency: Tensor) -> Tensor:
    return 2595.0 * torch.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mel: Tensor) -> Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def _make_mel_filterbank(
    *,
    sample_rate: int,
    fft_size: int,
    num_mel_bins: int,
    device: torch.device,
) -> Tensor:
    frequencies = torch.fft.rfftfreq(
        fft_size, d=1.0 / sample_rate, device=device
    )
    mel_edges = torch.linspace(
        _hz_to_mel(torch.tensor(0.0, device=device)),
        _hz_to_mel(torch.tensor(sample_rate / 2, device=device)),
        num_mel_bins + 2,
        device=device,
    )
    hz_edges = _mel_to_hz(mel_edges)
    lower = hz_edges[:-2, None]
    center = hz_edges[1:-1, None]
    upper = hz_edges[2:, None]
    rising = (frequencies[None, :] - lower) / (center - lower)
    falling = (upper - frequencies[None, :]) / (upper - center)
    return torch.clamp(torch.minimum(rising, falling), min=0.0)


class LogMelFrontend(nn.Module):
    """Causal 80-bin log-mel frontend with optional fixed normalization stats."""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        num_mel_bins: int = NUM_MEL_BINS,
        mean: Tensor | None = None,
        std: Tensor | None = None,
    ) -> None:
        super().__init__()
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"sample_rate must be {SAMPLE_RATE}")
        if num_mel_bins != NUM_MEL_BINS:
            raise ValueError(f"num_mel_bins must be {NUM_MEL_BINS}")
        self.sample_rate = sample_rate
        self.register_buffer("window", torch.hann_window(FRAME_LENGTH))
        self.register_buffer(
            "mel_filterbank",
            _make_mel_filterbank(
                sample_rate=sample_rate,
                fft_size=FFT_SIZE,
                num_mel_bins=num_mel_bins,
                device=torch.device("cpu"),
            ),
        )
        self.register_buffer(
            "mean",
            torch.zeros(num_mel_bins) if mean is None else mean.reshape(num_mel_bins),
        )
        self.register_buffer(
            "std",
            torch.ones(num_mel_bins) if std is None else std.reshape(num_mel_bins),
        )
        self.has_statistics = mean is not None and std is not None
        if self.has_statistics and torch.any(self.std <= 0).item():
            raise ValueError("mel normalization standard deviations must be positive")

    def set_statistics(self, mean: Tensor, std: Tensor) -> None:
        mean = mean.to(device=self.mean.device, dtype=self.mean.dtype).reshape(-1)
        std = std.to(device=self.std.device, dtype=self.std.dtype).reshape(-1)
        if mean.shape != (NUM_MEL_BINS,) or std.shape != (NUM_MEL_BINS,):
            raise ValueError(f"mel statistics must have shape ({NUM_MEL_BINS},)")
        if not torch.isfinite(mean).all().item() or not torch.isfinite(std).all().item():
            raise ValueError("mel statistics must be finite")
        if torch.any(std <= 0).item():
            raise ValueError("mel normalization standard deviations must be positive")
        self.mean.copy_(mean)
        self.std.copy_(std)
        self.has_statistics = True

    def forward(self, waveforms: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        if waveforms.ndim != 2:
            raise ValueError("waveforms must have shape (batch, samples)")
        if lengths.shape != (waveforms.shape[0],):
            raise ValueError("lengths must have shape (batch,)")
        if waveforms.shape[0] == 0:
            raise ValueError("waveforms must contain at least one example")
        if torch.any(lengths > waveforms.shape[1]).item():
            raise ValueError("waveform lengths must be positive and within the padded batch")
        if torch.any(lengths <= 0).item():
            raise ValueError("each waveform must contain at least one sample")

        batch_features: list[Tensor] = []
        for waveform, length in zip(waveforms, lengths.tolist(), strict=True):
            valid_audio = waveform[:length]
            left_padded = F.pad(valid_audio, (FRAME_LENGTH - 1, 0))
            spectrum = torch.stft(
                left_padded,
                n_fft=FFT_SIZE,
                hop_length=FRAME_HOP,
                win_length=FRAME_LENGTH,
                window=self.window.to(dtype=waveforms.dtype),
                center=False,
                return_complex=True,
            )
            power = spectrum.abs().square()
            mel_power = self.mel_filterbank.to(dtype=power.dtype) @ power
            log_mel = torch.log(torch.clamp(mel_power, min=MIN_LOG_ENERGY))
            batch_features.append(log_mel.transpose(0, 1))

        frame_lengths = torch.tensor(
            [features.shape[0] for features in batch_features],
            dtype=torch.long,
            device=waveforms.device,
        )
        padded = nn.utils.rnn.pad_sequence(
            batch_features, batch_first=True, padding_value=0.0
        )
        positions = torch.arange(padded.shape[1], device=waveforms.device)
        frame_mask = positions.unsqueeze(0) < frame_lengths.unsqueeze(1)
        normalized = (padded - self.mean) / self.std
        normalized = normalized.masked_fill(~frame_mask.unsqueeze(-1), 0.0)
        return normalized, frame_mask


def compute_mel_statistics(
    frontend: LogMelFrontend,
    batches: Iterable[dict[str, Any]],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Fit per-bin population statistics from waveform batches."""
    total = torch.zeros(NUM_MEL_BINS, dtype=torch.float64, device=device)
    total_squared = torch.zeros_like(total)
    frame_count = 0
    for batch in batches:
        waveforms = batch["audio"].to(device=device, dtype=torch.float32)
        lengths = batch["audio_lengths"].to(device=device)
        features, mask = frontend(waveforms, lengths)
        valid = features[mask].to(torch.float64)
        total += valid.sum(dim=0)
        total_squared += valid.square().sum(dim=0)
        frame_count += valid.shape[0]
    if frame_count == 0:
        raise ValueError("cannot compute mel statistics from an empty loader")
    mean = total / frame_count
    variance = torch.clamp(total_squared / frame_count - mean.square(), min=0.0)
    std = torch.sqrt(variance).clamp_min(1e-5)
    return mean.to(torch.float32), std.to(torch.float32)


def save_mel_statistics(path: str | Path, mean: Tensor, std: Tensor) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        mean=mean.detach().cpu().numpy().astype(np.float32),
        std=std.detach().cpu().numpy().astype(np.float32),
    )


def load_mel_statistics(path: str | Path) -> tuple[Tensor, Tensor]:
    with np.load(Path(path), allow_pickle=False) as statistics:
        if set(statistics.files) != {"mean", "std"}:
            raise ValueError("mel statistics file must contain only mean and std arrays")
        mean = torch.from_numpy(np.asarray(statistics["mean"], dtype=np.float32).copy())
        std = torch.from_numpy(np.asarray(statistics["std"], dtype=np.float32).copy())
    if mean.shape != (NUM_MEL_BINS,) or std.shape != (NUM_MEL_BINS,):
        raise ValueError(f"mel statistics arrays must have shape ({NUM_MEL_BINS},)")
    if not torch.isfinite(mean).all().item() or not torch.isfinite(std).all().item():
        raise ValueError("mel statistics must be finite")
    if torch.any(std <= 0).item():
        raise ValueError("mel normalization standard deviations must be positive")
    return mean, std
