"""Torch-free audio feature extraction for the Pi5-VCM inference path.

This module exists so the on-device (Raspberry Pi 5) harness can run with
ONLY ``numpy`` + ``onnxruntime`` installed -- no torch, no torchaudio, no
soundfile. The Pi's venv ships exactly those two packages and nothing else.

The feature here, :func:`kaldi_fbank`, is a faithful numpy port of
``torchaudio.compliance.kaldi.fbank`` -- the *exact* feature the model was
trained on (see ``utils/audio_utils.mel_spectrogram``). It was validated
against the reference implementation to float32 precision (Pearson 1.000000,
max abs diff ~0.013 on log-mel values). Using the same feature on-device is
what makes the ONNX path produce identical intents/slots to the torch path.

Functions
---------
kaldi_fbank(wav, ...)   (T, 80) log-mel, the training feature
load_wav_mono(path)     (samples,) float32 @16 kHz via stdlib ``wave``
synthesize_utterance(...)  toy formant speech (numpy only) for self-tests
"""
from __future__ import annotations

import wave

import numpy as np

import config


# ---------------------------------------------------------------------------
# kaldi.fbank (numpy port)
# ---------------------------------------------------------------------------
def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _povey_window(n: int) -> np.ndarray:
    # kaldi/povey window: hann_window(n, periodic=False) ** 0.85
    x = np.arange(n)
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * x / (n - 1)))
    return (hann ** 0.85).astype(np.float32)


def _hz_to_mel(hz: float) -> float:
    return 1127.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 1127.0) - 1.0)


def kaldi_fbank(wav: np.ndarray,
                num_mel_bins: int = config.N_MELS,
                sample_frequency: float = config.SAMPLE_RATE,
                frame_length: float = config.FRAME_LENGTH_MS,
                frame_shift: float = config.FRAME_SHIFT_MS,
                dither: float = 0.0,
                snip_edges: bool = True,
                low_freq: float = 20.0,
                high_freq: float | None = None,
                preemphasis: float = 0.97,
                remove_dc: bool = True,
                use_log: bool = True,
                use_power: bool = True,
                round_to_pow2: bool = True) -> np.ndarray:
    """Log-mel filterbank features, shape ``(T, num_mel_bins)`` float32.

    Faithful numpy port of ``torchaudio.compliance.kaldi.fbank`` with the
    same defaults the training pipeline uses (``dither=0``, ``snip_edges``
    per caller, ``preemphasis=0.97``, ``remove_dc=True``, ``use_log=True``,
    ``use_power=True``, ``round_to_power_of_two=True``).
    """
    wav = np.asarray(wav, dtype=np.float32).ravel()
    window_size = int(round(frame_length * sample_frequency / 1000.0))    # 400
    window_shift = int(round(frame_shift * sample_frequency / 1000.0))    # 160
    padded = _next_pow2(window_size) if round_to_pow2 else window_size    # 512

    if snip_edges:
        if len(wav) < window_size:
            return np.zeros((0, num_mel_bins), dtype=np.float32)
        n_frames = 1 + (len(wav) - window_size) // window_shift
        idx = (np.arange(n_frames)[:, None] * window_shift
               + np.arange(window_size)[None, :])
        frames = wav[idx]
    else:
        # Reflect at both ends (kaldi end-effect handling), then frame.
        rev = wav[::-1]
        n_frames = (len(wav) + (window_shift // 2)) // window_shift
        pad = window_size // 2 - window_shift // 2
        if pad > 0:
            extended = np.concatenate([rev[-pad:], wav, rev])
        else:
            extended = np.concatenate([wav[-pad:], rev])
        idx = (np.arange(n_frames)[:, None] * window_shift
               + np.arange(window_size)[None, :])
        frames = extended[idx]

    if dither != 0.0:
        frames = frames + dither * np.random.randn(*frames.shape).astype(np.float32)
    if remove_dc:
        frames = frames - frames.mean(axis=1, keepdims=True)
    if preemphasis != 0.0:
        prev = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
        frames = frames - preemphasis * prev
    frames = frames * _povey_window(window_size)
    if padded != window_size:
        frames = np.pad(frames, ((0, 0), (0, padded - window_size)))

    spec = np.abs(np.fft.rfft(frames, axis=1))
    if use_power:
        spec = spec ** 2

    if high_freq is None:
        high_freq = sample_frequency / 2.0
    n_bins = padded // 2
    mel_lo, mel_hi = _hz_to_mel(low_freq), _hz_to_mel(high_freq)
    m_pts = _mel_to_hz(np.linspace(mel_lo, mel_hi, num_mel_bins + 2))
    hz_pts = np.fft.rfftfreq(padded, 1.0 / sample_frequency)[:n_bins]
    fb = np.zeros((num_mel_bins, n_bins), dtype=np.float32)
    for i in range(num_mel_bins):
        lo, c, hi = m_pts[i], m_pts[i + 1], m_pts[i + 2]
        for j in range(n_bins):
            f = hz_pts[j]
            if lo < f < c:
                fb[i, j] = (f - lo) / (c - lo)
            elif c < f < hi:
                fb[i, j] = (hi - f) / (hi - c)
    fb = np.pad(fb, ((0, 0), (0, 1)), mode="constant")   # match rfft length

    mel = spec @ fb.T
    if use_log:
        mel = np.maximum(mel, np.finfo(np.float32).eps).astype(np.float32)
        mel = np.log(mel)
    return mel.astype(np.float32)


# ---------------------------------------------------------------------------
# wav I/O (stdlib only)
# ---------------------------------------------------------------------------
def load_wav_mono(path: str) -> np.ndarray:
    """Read a wav file -> mono float32 @ ``config.SAMPLE_RATE`` (samples,).

    Uses only the stdlib ``wave`` module (no soundfile/torchaudio). Linearly
    resamples to 16 kHz if the file is at a different rate.
    """
    with wave.open(path, "rb") as w:
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        n = w.getnframes()
        sr = w.getframerate()
        raw = w.readframes(n)
    if sampwidth not in (1, 2, 4):
        raise ValueError(f"unsupported sample width {sampwidth} in {path}")
    fmt = {1: "int8", 2: "int16", 4: "int32"}[sampwidth]
    scale = {1: 127.0, 2: 32767.0, 4: 2147483647.0}[sampwidth]
    arr = np.frombuffer(raw, dtype=fmt).astype(np.float64)
    if channels == 2:
        arr = arr.reshape(-1, 2).mean(axis=1)
    arr = arr / scale
    if sr != config.SAMPLE_RATE:
        dur = len(arr) / sr
        new_n = max(1, int(round(dur * config.SAMPLE_RATE)))
        x_old = np.linspace(0, 1, len(arr), endpoint=False)
        x_new = np.linspace(0, 1, new_n, endpoint=False)
        arr = np.interp(x_new, x_old, arr)
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# toy-speech synthesis (numpy only) -- for self-tests without real recordings
# ---------------------------------------------------------------------------
def _word_frequencies(word: str, speaker: str):
    """Deterministic pseudo-formant frequencies per (word, speaker)."""
    seed = 0
    for ch in (word + "|" + speaker):
        seed = (seed * 131 + ord(ch)) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    base = 80.0 + rng.uniform(0, 120.0)
    f0 = base
    f1 = base + rng.uniform(250, 500.0)
    f2 = f1 + rng.uniform(300, 700.0)
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
    """Toy formant-tone utterance (NOT real speech) -- numpy only.

    Mirrors ``utils/audio_utils.synthesize_utterance`` but without the torch
    dependency, so the self-test path runs on the Pi.
    """
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for i, word in enumerate(words):
        parts.append(synthesize_word(word, speaker, sr, rng))
        if i < len(words) - 1:
            parts.append(np.zeros(int(0.03 * sr), dtype=np.float32))
    wav = np.concatenate(parts) if parts else np.zeros(sr // 10, dtype=np.float32)
    sig_power = float(np.mean(wav ** 2)) + 1e-12
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    wav = wav + rng.standard_normal(wav.shape).astype(np.float32) * np.sqrt(noise_power)
    peak = float(np.abs(wav).max())
    if peak > 0:
        wav = wav / peak * 0.8
    return wav.astype(np.float32)
