#!/usr/bin/env python3
"""Synthesize the two expiry sounds for the voice assistant.

Produces (into rpi5/assets/replies/, 22050 Hz mono 16-bit to match the
existing assets):

  timer_ding.wav  -- a gentle rising triple-ding. "Your time is up."
  alarm_ring.wav  -- a loud, urgent repeating buzz. "Wake up."

Pure stdlib (math/wave/struct). No numpy, no scipy, no ffmpeg.

Design intent
-------------
The two must be unmistakably different from each other AND from the soft
ack_beep used for wake acknowledgement:
  * timer = bright, short, ascending, light amplitude -> calm
  * alarm = low, harsh, repeating, full amplitude     -> urgent
"""
from __future__ import annotations

import math
import os
import struct
import wave

SR = 22050          # sample rate (matches existing assets)
# run.py defaults are relative to cwd, and the harness cd's to me2/ (see
# run_pi5_harness.sh: ROOT=me2/, "cd ${ROOT}"). So the live assets dir is
# me2/assets/replies/, NOT rpi5/assets/replies/.
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "replies")


def _env(t: float, dur: float, attack: float = 0.005, release: float = 0.04) -> float:
    """Simple linear attack/release envelope in [0, 1]."""
    if t < 0:
        return 0.0
    if t < attack:
        return t / attack
    rel_start = dur - release
    if t > rel_start:
        return max(0.0, (dur - t) / release)
    return 1.0


def _tone(freq: float, dur: float, sr: int = SR) -> list[float]:
    """One sine tone with an attack/release envelope."""
    n = int(sr * dur)
    out = []
    for i in range(n):
        t = i / sr
        s = math.sin(2 * math.pi * freq * t)
        out.append(s * _env(t, dur))
    return out


def _buzz(freq: float, dur: float, sr: int = SR, harmonics: tuple[float, float] = (1.0, 0.5)) -> list[float]:
    """Harsh buzz: fundamental + a strong overtone, giving a buzzer timbre."""
    n = int(sr * dur)
    out = []
    for i in range(n):
        t = i / sr
        s = harmonics[0] * math.sin(2 * math.pi * freq * t)
        s += harmonics[1] * math.sin(2 * math.pi * freq * 2.0 * t)
        # normalize the two-part sum roughly to unit peak
        s /= (harmonics[0] + harmonics[1])
        out.append(s * _env(t, dur, attack=0.003, release=0.02))
    return out


def _concat(parts: list[list[float]], gap: float = 0.0) -> list[float]:
    out = list(parts[0])
    gap_n = int(SR * gap)
    for p in parts[1:]:
        out.extend([0.0] * gap_n)
        out.extend(p)
    return out


def _normalize(samples: list[float], peak: float = 0.9) -> list[int]:
    m = max((abs(s) for s in samples), default=0.0)
    scale = peak / m if m > 0 else 0.0
    return [int(max(-1.0, min(1.0, s * scale)) * 32767) for s in samples]


def _write(path: str, samples: list[int]) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(struct.pack("<%dh" % len(samples), *samples))
    print(f"wrote {os.path.relpath(path)}  ({len(samples)/SR:.2f}s)")


def make_timer_ding() -> None:
    """Gentle rising triple-ding: C6 -> E6 -> G6, short and light."""
    notes = [1046.50, 1318.51, 1568.00]   # C6, E6, G6 (major triad, ascending)
    parts = [_tone(f, 0.16) for f in notes]
    samples = _concat(parts, gap=0.06)
    _write(os.path.join(OUT_DIR, "timer_ding.wav"), _normalize(samples, peak=0.55))


def make_alarm_ring() -> None:
    """Urgent repeating buzz: four short low buzzes with brief gaps."""
    buzzes = [_buzz(660.0, 0.28) for _ in range(4)]
    samples = _concat(buzzes, gap=0.12)
    _write(os.path.join(OUT_DIR, "alarm_ring.wav"), _normalize(samples, peak=0.9))


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    make_timer_ding()
    make_alarm_ring()
    print("done.")
