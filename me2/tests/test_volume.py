"""Unit tests for system output-volume control (rpi5/volume.py).

Covers the demo capture/restore contract, the step/mute/unmute controls, and
the new Echo-style duck/unduck behaviour that lowers the master volume while
the assistant produces audio so the microphone can still catch the wake word.

No real audio is touched: the controller's get/set are injected as callables
that record levels, mirroring the pattern in ``tests/test_media.py``.
"""
from __future__ import annotations

import unittest

from rpi5.volume import (DUCK_LEVEL, SAFE_FALLBACK_LEVEL, VolumeController)


class FakeVolume:
    """In-memory stand-in for the OS output volume."""

    def __init__(self, start: int = 80):
        self.level = start
        self.sets: list[int] = []

    def get(self) -> int:
        return self.level

    def set(self, level: int) -> None:
        self.level = level
        self.sets.append(level)


def make_ctrl(start: int = 80) -> tuple[VolumeController, FakeVolume]:
    fake = FakeVolume(start)
    ctrl = VolumeController(backend="none", get=fake.get, set=fake.set)
    return ctrl, fake


class DuckUnduckTests(unittest.TestCase):
    def test_duck_lowers_to_duck_level(self):
        ctrl, fake = make_ctrl(start=90)
        applied = ctrl.duck()
        self.assertEqual(applied, DUCK_LEVEL)
        self.assertEqual(fake.level, DUCK_LEVEL)

    def test_unduck_restores_pre_duck_level(self):
        ctrl, fake = make_ctrl(start=90)
        ctrl.duck()
        restored = ctrl.unduck()
        self.assertEqual(restored, 90)
        self.assertEqual(fake.level, 90)

    def test_nested_duck_remembers_first_level(self):
        ctrl, fake = make_ctrl(start=90)
        ctrl.duck()
        ctrl.duck()  # second duck must NOT compound the dip
        restored = ctrl.unduck()
        self.assertEqual(restored, 90)
        self.assertEqual(fake.level, 90)

    def test_unduck_without_duck_is_noop(self):
        ctrl, fake = make_ctrl(start=70)
        self.assertIsNone(ctrl.unduck())
        self.assertEqual(fake.level, 70)  # untouched

    def test_duck_when_level_unreadable_uses_safe_fallback(self):
        # get() raises -> duck still applies the duck level and remembers the
        # safe fallback so unduck does not crash.
        fake = FakeVolume(50)

        def boom() -> int:
            raise RuntimeError("no backend")

        ctrl = VolumeController(backend="none", get=boom, set=fake.set)
        applied = ctrl.duck()
        self.assertEqual(applied, DUCK_LEVEL)
        self.assertEqual(fake.level, DUCK_LEVEL)
        restored = ctrl.unduck()
        self.assertEqual(restored, SAFE_FALLBACK_LEVEL)

    def test_restore_clears_duck_state(self):
        ctrl, fake = make_ctrl(start=90)
        ctrl.begin()  # capture 90
        ctrl.duck()
        ctrl.restore()
        self.assertEqual(fake.level, 90)
        # After restore the duck is gone; unduck must be a no-op.
        self.assertIsNone(ctrl.unduck())
        self.assertEqual(fake.level, 90)


class UnmuteFallbackTests(unittest.TestCase):
    def test_unmute_uses_captured_level(self):
        ctrl, fake = make_ctrl(start=60)
        ctrl.begin()  # capture 60
        ctrl.mute()
        self.assertEqual(fake.level, 0)
        self.assertEqual(ctrl.unmute(), 60)
        self.assertEqual(fake.level, 60)

    def test_unmute_without_capture_never_hits_max(self):
        # The historical bug: unmute with no captured level jumped to 100%.
        # It must fall back to the safe level (50%), never full scale.
        ctrl, fake = make_ctrl(start=30)
        # no begin() -> _captured is None
        ctrl.mute()
        self.assertEqual(ctrl.unmute(), SAFE_FALLBACK_LEVEL)
        self.assertEqual(fake.level, SAFE_FALLBACK_LEVEL)
        self.assertNotEqual(fake.level, 100)


class DemoContractTests(unittest.TestCase):
    def test_begin_end_roundtrip(self):
        ctrl, fake = make_ctrl(start=75)
        self.assertEqual(ctrl.begin(), 75)
        ctrl.set(20)
        self.assertTrue(ctrl.end())
        self.assertEqual(fake.level, 75)

    def test_end_is_idempotent(self):
        ctrl, fake = make_ctrl(start=75)
        ctrl.begin()
        self.assertTrue(ctrl.end())
        self.assertTrue(ctrl.end())
        self.assertEqual(fake.level, 75)


class StepMuteTests(unittest.TestCase):
    def test_step_clamps_and_moves(self):
        ctrl, fake = make_ctrl(start=95)
        self.assertEqual(ctrl.step(10), 100)  # clamped at top
        self.assertEqual(ctrl.step(-10), 90)

    def test_mute_preserves_for_unmute(self):
        ctrl, fake = make_ctrl(start=85)
        ctrl.begin()
        ctrl.mute()
        self.assertEqual(fake.level, 0)
        self.assertEqual(ctrl.unmute(), 85)


if __name__ == "__main__":
    unittest.main()
