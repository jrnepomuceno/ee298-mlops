"""Tests for the microphone capture/VAD helpers (no hardware required)."""
from __future__ import annotations

import math
import unittest

import numpy as np

from rpi5.audio import EnergyVAD, VADConfig, drain_until_quiet


def loud(rms: float = 0.2, size: int = 480) -> np.ndarray:
    """A frame whose RMS is roughly ``rms``."""
    return np.full(size, rms, dtype=np.float32)


def quiet(size: int = 480) -> np.ndarray:
    return np.zeros(size, dtype=np.float32)


class DrainUntilQuietTests(unittest.TestCase):
    """``drain_until_quiet`` is what stops the assistant re-hearing its reply."""

    def test_returns_as_soon_as_room_is_quiet(self):
        settings = VADConfig()
        frames = [loud()] + [quiet() for _ in range(settings.silence_frames + 3)]

        def get_frame():
            return frames.pop(0)

        drain_until_quiet(get_frame, settings, min_s=0.0, max_s=1.0)

        # It stopped once the room went quiet instead of burning the whole
        # budget, so the trailing quiet frames were never read.
        self.assertEqual(len(frames), 3)

    def test_keeps_draining_while_assistant_is_still_speaking(self):
        """Regression: a loud TTS tail outlives a fixed cooldown.

        The old code discarded frames for a fixed ``cooldown_s`` and returned.
        With a 1.5 s reply the tail is still buffered, the VAD re-triggers on
        the assistant's own voice, and the command runs a second time. Draining
        on the signal must consume the whole loud run.
        """
        settings = VADConfig()
        loud_run = [loud() for _ in range(40)]   # ~1.2 s of assistant speech
        frames = list(loud_run) + [quiet() for _ in range(settings.silence_frames + 2)]
        consumed = []

        def get_frame():
            frame = frames.pop(0)
            consumed.append(frame)
            return frame

        drain_until_quiet(get_frame, settings, min_s=0.0, max_s=5.0)

        self.assertEqual(len(consumed), len(loud_run) + settings.silence_frames)

    def test_is_bounded_by_max_s(self):
        """A permanently loud room must not wedge the capture loop."""
        settings = VADConfig()
        state = {"n": 0}

        def get_frame():
            state["n"] += 1
            return loud()

        drain_until_quiet(get_frame, settings, min_s=0.0, max_s=0.2)

        # Bounded by frames as well as the clock, so even a source that returns
        # instantly cannot spin.
        cap = math.ceil(0.2 * 1000.0 / settings.frame_ms) + settings.silence_frames
        self.assertLessEqual(state["n"], cap)

    def test_min_s_is_honoured(self):
        settings = VADConfig()
        state = {"n": 0}

        def get_frame():
            state["n"] += 1
            return quiet()

        drain_until_quiet(get_frame, settings, min_s=0.1, max_s=5.0)

        # Quiet from the start, but the floor still forces a short drain.
        self.assertGreaterEqual(state["n"], 3)


class EnergyVadTests(unittest.TestCase):
    def test_utterance_needs_speech_then_silence(self):
        vad = EnergyVAD(VADConfig())

        self.assertIsNone(vad.accept(loud()))
        self.assertIsNone(vad.accept(loud()))
        self.assertTrue(vad.active)

        for _ in range(VADConfig().silence_frames):
            utterance = vad.accept(quiet())

        self.assertIsNotNone(utterance)
        self.assertFalse(vad.active)

    def test_reset_clears_partial_utterance(self):
        vad = EnergyVAD(VADConfig())
        vad.accept(loud())
        vad.accept(loud())
        self.assertTrue(vad.active)

        vad.reset()

        self.assertFalse(vad.active)
        self.assertIsNone(vad.flush())


if __name__ == "__main__":
    unittest.main()
