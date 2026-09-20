"""Audio I/O, feature extraction, and toy-speech synthesis.

Only torch / torchaudio / numpy are used, so this runs on the Pi's venv.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
import torchaudio

import config


def load_wav_mono(path, sample_rate: int = config.SAMPLE_RATE) -> torch.Tensor:
    """Load a wav file as a float32 mono tensor at `sample_rate`."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.squeeze(0).float()


def mel_spectrogram(wav: torch.Tensor,
                    sample_rate: int = config.SAMPLE_RATE,
                    n_mels: int = config.N_MELS) -> torch.Tensor:
    """Log-mel features, shape (T, n_mels). kaldi.fbank already returns log."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    fbank = torchaudio.compliance.kaldi.fbank(
        wav,
        num_mel_bins=n_mels,
        sample_frequency=sample_rate,
        frame_length=config.FRAME_LENGTH_MS,
        frame_shift=config.FRAME_SHIFT_MS,
        dither=0.0,
        snip_edges=False,
    )
    return fbank  # (T, n_mels)


def pad_or_trim(mel: torch.Tensor, max_frames: int) -> torch.Tensor:
    """Right-pad (zero) or trim to at most `max_frames` rows."""
    if mel.shape[0] >= max_frames:
        return mel[:max_frames]
    pad = torch.zeros(max_frames - mel.shape[0], mel.shape[1])
    return torch.cat([mel, pad], dim=0)


def spec_augment(mel: torch.Tensor,
                 num_freq_masks: int = 2,
                 num_time_masks: int = 2,
                 max_freq_mask: int = 27,
                 max_time_mask: int = 40,
                 p: float = 0.5) -> torch.Tensor:
    """SpecAugment: random frequency/time masking (returns a copy)."""
    out = mel.clone()
    if torch.rand(()) >= p:
        return out
    t, f = out.shape
    for _ in range(num_freq_masks):
        width = int(torch.randint(0, max_freq_mask + 1, (1,)).item())
        if 0 < width < f:
            start = int(torch.randint(0, f - width + 1, (1,)).item())
            out[start:start + width, :] = 0.0
    for _ in range(num_time_masks):
        width = int(torch.randint(0, max_time_mask + 1, (1,)).item())
        if 0 < width < t:
            start = int(torch.randint(0, t - width + 1, (1,)).item())
            out[:, start:start + width] = 0.0
    return out


def add_noise(wav: torch.Tensor, snr_db: float,
              rng: np.random.Generator) -> torch.Tensor:
    """Add white Gaussian noise at a target SNR (dB)."""
    power = wav.pow(2).mean().clamp_min(1e-10)
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = torch.from_numpy(
        rng.standard_normal(wav.numel()).astype(np.float32))
    return wav + noise * noise_power.sqrt()


# ------------------------------------------------------- toy speech synthesis --

def _word_frequencies(word: str, speaker: str) -> tuple[float, float, float]:
    """Deterministic pseudo-formant frequencies for a (word, speaker) pair."""
    seed = int(hashlib.sha256(f"{speaker}|{word}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    f0 = float(rng.uniform(80, 220))
    f1 = float(rng.uniform(200, 900))
    f2 = float(rng.uniform(900, 2800))
    return f0, f1, f2


def synthesize_word(word: str, speaker: str, sr: int,
                    rng: np.random.Generator) -> np.ndarray:
    f0, f1, f2 = _word_frequencies(word, speaker)
    dur = float(rng.uniform(0.14, 0.22))
    n = max(int(dur * sr), 1)
    t = np.arange(n) / sr
    env = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / dur))
    sig = (0.55 * np.sin(2.0 * np.pi * f1 * t)
           + 0.35 * np.sin(2.0 * np.pi * f2 * t)
           + 0.10 * np.sin(2.0 * np.pi * f0 * t))
    return (sig * env).astype(np.float32)


def synthesize_utterance(words: list[str], speaker: str,
                         sr: int = config.SAMPLE_RATE,
                         snr_db: float = 20.0,
                         seed: int = 0) -> np.ndarray:
    """Render a word list as a toy utterance: formant tones + gaps + noise.

    This is NOT real speech. It exists so the dataset/dataloader/model/
    training pipeline can be developed and smoke-tested before real
    recordings arrive.
    """
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for i, word in enumerate(words):
        parts.append(synthesize_word(word, speaker, sr, rng))
        if i < len(words) - 1:
            parts.append(np.zeros(int(0.03 * sr), dtype=np.float32))
    wav = np.concatenate(parts) if parts else np.zeros(sr // 10, dtype=np.float32)
    wav = add_noise(torch.from_numpy(wav), snr_db, rng).numpy()
    peak = float(np.abs(wav).max())
    if peak > 0:
        wav = wav / peak * 0.8
    return wav.astype(np.float32)
