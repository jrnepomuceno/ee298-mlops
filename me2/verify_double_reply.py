"""A/B verification that the post-reply drain stops the double reply.

Simulates microphone_utterances with a *time-paced* frame source (the real
pw-record/sounddevice path blocks per frame, so a wall-clock cooldown covers a
known number of frames). Script:

    user speech -> [assistant reply, audible to the mic] -> room goes quiet

With the old fixed cooldown the assistant's own reply is still buffered when the
cooldown expires, so the VAD re-triggers and a second utterance is yielded.
"""
from __future__ import annotations

import shutil
import time

import numpy as np

import rpi5.audio as audio


FRAME_SAMPLES_48K = 1440  # 30 ms @ 48 kHz, what _pipewire_capture yields


def frame(rms: float) -> np.ndarray:
    return np.full(FRAME_SAMPLES_48K, rms, dtype=np.float32)


def build_script() -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    frames += [frame(0.2)] * 10          # user speaking
    frames += [frame(0.0)] * 12          # user stops -> utterance 1 closes
    frames += [frame(0.2)] * 40          # assistant reply, captured by the mic
    frames += [frame(0.0)] * 40          # room finally quiet
    return frames


def run(label: str) -> int:
    script = build_script()
    index = {"i": 0}

    def fake_pipewire_capture(settings):
        for f in script:
            # Pace frames in real time, exactly like a blocking capture read.
            time.sleep(settings.frame_ms / 1000.0)
            index["i"] += 1
            yield f

    audio._pipewire_capture = fake_pipewire_capture
    audio.shutil = shutil
    shutil.which = lambda *a, **k: "/usr/bin/pw-record"

    yielded = 0
    for _utt in audio.microphone_utterances(cooldown_s=1.0, max_drain_s=3.0):
        yielded += 1
        # The consumer does inference + synchronous TTS playback here; the mic
        # keeps capturing the assistant's own voice meanwhile (already scripted).
    print(f"{label}: utterances yielded = {yielded}")
    return yielded


def old_drain(get_frame, settings, min_s=0.0, max_s=3.0):
    """The previous behaviour: discard for a fixed cooldown, then move on."""
    deadline = time.monotonic() + min_s
    while time.monotonic() < deadline:
        get_frame()


if __name__ == "__main__":
    print(f"frames per 1.0s cooldown at 30ms/frame = {1000 // 30}")
    print(f"assistant reply buffered = 40 frames (1.2s) -- outlives the cooldown\n")

    new = run("NEW (drain_until_quiet)")

    audio.drain_until_quiet = old_drain
    old = run("OLD (fixed cooldown)  ")

    print()
    assert new == 1, f"expected exactly one utterance, got {new}"
    assert old == 2, f"expected the old code to double-fire, got {old}"
    print(f"PASS: old code replies twice ({old}), fixed code replies once ({new})")
