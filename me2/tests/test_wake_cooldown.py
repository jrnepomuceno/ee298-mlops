"""Regression tests for the wake-word inter-detection cooldown.

The "double Alexa" bug: after the detector fires on the user's "Alexa", the
same audio (or its room echo) can re-latch the detector within a few hundred
milliseconds, producing a second spurious wake and a duplicate interaction.

The fix adds a timestamp-based cooldown: after a detection fires, accepts()
returns False for cooldown_s seconds regardless of model scores. reset()
re-arms the cooldown so the next legitimate detection is allowed immediately.

These tests use a mocked openwakeword model (no real inference) so they run
fast and deterministically on any platform.
"""
from __future__ import annotations

import time
import unittest
from unittest import mock

import numpy as np

from rpi5.wakeword import OpenWakeWordDetector


def _make_detector(cooldown_s: float = 2.0, threshold: float = 0.7):
    """Construct an OpenWakeWordDetector with a mocked model."""
    with mock.patch("openwakeword.get_pretrained_model_paths",
                    return_value=["/fake/alexa.onnx"]), \
         mock.patch("openwakeword.model.Model") as MockModel:
        det = OpenWakeWordDetector(
            model_name="alexa",
            threshold=threshold,
            cooldown_s=cooldown_s,
        )
    return det


def _frame(rate: int = 16_000, ms: int = 80) -> np.ndarray:
    """A random-noise frame sized to one detector window (1280 samples)."""
    n = rate * ms // 1000
    rng = np.random.default_rng(42)
    return rng.uniform(-0.5, 0.5, size=n).astype(np.float32)


class WakeCooldownTests(unittest.TestCase):
    """The detector must not re-fire within cooldown_s of the previous hit."""

    def test_cooldown_suppresses_immediate_refire(self):
        """After a detection, accepts() must return False for cooldown_s."""
        det = _make_detector(cooldown_s=2.0)
        # Mock the model to always return a high score (above threshold).
        det.model.predict = lambda window: {"alexa": 0.95}
        det.model.reset = lambda: None

        # Feed windows until the first detection fires (needs 2 consecutive).
        fired = False
        for _ in range(10):
            if det.accepts(_frame()):
                fired = True
                break
        self.assertTrue(fired, "detector should fire on sustained high scores")

        # Immediately after firing, the cooldown must suppress further hits
        # even though the model still scores above threshold.
        for _ in range(20):
            self.assertFalse(det.accepts(_frame()),
                             "detector must NOT re-fire during cooldown")

    def test_cooldown_expires_and_allows_next_detection(self):
        """After cooldown_s elapses, a new detection is allowed."""
        det = _make_detector(cooldown_s=0.1)  # very short for test speed
        det.model.predict = lambda window: {"alexa": 0.95}
        det.model.reset = lambda: None

        # First detection.
        for _ in range(10):
            if det.accepts(_frame()):
                break

        # Wait past the cooldown.
        time.sleep(0.15)

        # Now a new detection should be possible.
        fired_again = False
        for _ in range(10):
            if det.accepts(_frame()):
                fired_again = True
                break
        self.assertTrue(fired_again,
                        "detector should fire again after cooldown expires")

    def test_reset_rearms_cooldown(self):
        """reset() clears the cooldown so the next detection is immediate."""
        det = _make_detector(cooldown_s=5.0)  # long cooldown
        det.model.predict = lambda window: {"alexa": 0.95}
        det.model.reset = lambda: None

        # First detection.
        for _ in range(10):
            if det.accepts(_frame()):
                break

        # Simulate the state machine calling reset() after the ack.
        det.reset()

        # Despite the 5s cooldown, reset() re-armed the detector, so a new
        # detection should fire almost immediately.
        fired = False
        for _ in range(10):
            if det.accepts(_frame()):
                fired = True
                break
        self.assertTrue(fired,
                        "reset() must re-arm the cooldown for the next wake")

    def test_no_false_positive_on_low_scores(self):
        """Sub-threshold scores never fire, regardless of cooldown state."""
        det = _make_detector(cooldown_s=2.0)
        det.model.predict = lambda window: {"alexa": 0.1}
        det.model.reset = lambda: None

        for _ in range(50):
            self.assertFalse(det.accepts(_frame()))


if __name__ == "__main__":
    unittest.main()
