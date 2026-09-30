"""Portable WAV loading and deterministic Kaldi log-mel extraction."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio

SAMPLE_RATE = 16_000
N_MELS = 80
FRAME_LENGTH_MS = 25.0
FRAME_SHIFT_MS = 10.0


def load_wav_mono(path: str | Path) -> torch.Tensor:
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        sample_rate = stream.getframerate()
        sample_width = stream.getsampwidth()
        frame_count = stream.getnframes()
        if stream.getcomptype() != "NONE":
            raise ValueError(f"compressed WAV is not supported: {path}")
        raw = stream.readframes(frame_count)

    if sample_width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (packed[:, 0].astype(np.int32) |
                  (packed[:, 1].astype(np.int32) << 8) |
                  (packed[:, 2].astype(np.int32) << 16))
        values = (values ^ 0x800000) - 0x800000
        audio = values.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported PCM sample width {sample_width}: {path}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    return waveform


def mel_spectrogram(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim != 1:
        raise ValueError("expected mono waveform with shape (samples,)")
    return torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0),
        num_mel_bins=N_MELS,
        sample_frequency=SAMPLE_RATE,
        frame_length=FRAME_LENGTH_MS,
        frame_shift=FRAME_SHIFT_MS,
        dither=0.0,
        snip_edges=False,
    )
