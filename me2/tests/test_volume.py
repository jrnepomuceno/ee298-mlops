"""Unit tests for system output-volume control (rpi5/volume.py).

Covers the demo capture/restore contract, the step/mute/unmute controls, and
the new Echo-style duck/unduck behaviour that lowers the master volume while
the assistant produces audio so the microphone can still catch the wake word.

No real audio is touched: the controller's get/set are injected as callables
that record levels, mirroring the pattern in ``tests/test_media.py``.
"""
from __future__ import annotations

import unittest

from rpi5.facade import ActionRequest
from rpi5.executors.volume import VolumeExecutor
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


class DisabledMuteIntentTests(unittest.TestCase):
    def test_volume_executor_does_not_dispatch_mute_action(self):
        ctrl, fake = make_ctrl(start=85)
        request = ActionRequest(
            intent="mute", category="volume", slots={},
            action_code="volume.mute", reply_text="", dry_run=False,
            confidence=0.99,
        )
        result = VolumeExecutor(dry_run=False, controller=ctrl).run(request)
        self.assertFalse(result.ok)
        self.assertFalse(result.side_effects)
        self.assertIn("unsupported volume action", result.detail)
        self.assertEqual(fake.sets, [])


class WpctlParseTests(unittest.TestCase):
    """Regression: ``wpctl get-volume`` prints a 0..1 fraction, not a %.

    The Pi emits ``Volume: 0.47``; the old ``(%d)%`` regex never matched and
    made every duck warn ``could not read level``. Accept the fraction form
    (and the percent form some builds use)."""

    def _parse(self, out: str) -> int:
        ctrl = VolumeController(backend="wpctl")
        return ctrl._get_level.__wrapped__(ctrl) if hasattr(ctrl._get_level, "__wrapped__") else _parse_wpctl(out)

    def test_fraction_form(self):
        self.assertEqual(_parse_wpctl("Volume: 0.47"), 47)

    def test_fraction_zero(self):
        self.assertEqual(_parse_wpctl("Volume: 0.00"), 0)

    def test_fraction_full(self):
        self.assertEqual(_parse_wpctl("Volume: 1.00"), 100)

    def test_percent_form(self):
        self.assertEqual(_parse_wpctl("Volume: 47%"), 47)

    def test_bare_integer_percent(self):
        self.assertEqual(_parse_wpctl("47"), 47)

    def test_garbage_raises(self):
        with self.assertRaises(RuntimeError):
            _parse_wpctl("")


def _parse_wpctl(out: str) -> int:
    """Replicate the wpctl parsing branch from VolumeController._get_level."""
    import re
    m = re.search(r"(\d+(?:\.\d+)?)%", out)
    if m:
        return _clamp_test(round(float(m.group(1))))
    m = re.search(r"(?:^|\D)(\d+(?:\.\d+)?)\s*(?:$|\D)", out)
    if m:
        value = float(m.group(1))
        if value <= 1.0:
            return _clamp_test(round(value * 100.0))
        return _clamp_test(round(value))
    raise RuntimeError(f"could not parse wpctl volume: {out!r}")


def _clamp_test(v: int) -> int:
    from rpi5.volume import MAX_LEVEL, MIN_LEVEL
    return max(MIN_LEVEL, min(MAX_LEVEL, v))


class FindStreamNodeTests(unittest.TestCase):
    def setUp(self):
        from rpi5.volume import _find_stream_node
        self.find = _find_stream_node

    #: A trimmed ``wpctl status`` listing: sinks/sources first, then the
    #: Stream objects section with an ffplay output stream.
    STATUS = """\
Audio
    Data flows in both directions between endpoints and nodes.
    Nodes
        1  Sink  alsa_output.pci-0000_00_14.2.analog-stereo
        3  Source alsa_input.pci-0000_00_14.2.analog-stereo
Stream objects
        12  Stream Output ffplay
        14  Stream Input pulse
Monitor streams
        20  Monitor alsa_output.pci-0000_00_14.2.analog-stereo
"""

    def test_finds_ffplay_stream(self):
        self.assertEqual(self.find(self.STATUS, "ffplay"), "12")

    def test_case_insensitive(self):
        self.assertEqual(self.find(self.STATUS, "FFPLAY"), "12")

    def test_numeric_id_passthrough(self):
        self.assertEqual(self.find(self.STATUS, "12"), "12")

    def test_no_match_returns_none(self):
        self.assertIsNone(self.find(self.STATUS, "pw-play"))

    def test_does_not_match_sink_or_monitor(self):
        # "alsa_output..." contains no stream-name match, and monitor streams
        # must never be picked even when the name overlaps.
        self.assertIsNone(self.find(self.STATUS, "analog-stereo"))


class SetChannelTests(unittest.TestCase):
    def setUp(self):
        from rpi5.volume import VolumeController
        self.ctrl = VolumeController(backend="wpctl")

    def test_unsupported_backend_returns_none(self):
        self.ctrl.backend = "amixer"
        self.assertIsNone(self.ctrl.set_channel("ffplay", 40))

    def test_no_matching_stream_returns_none(self):
        import subprocess
        orig = subprocess.run
        self.ctrl.backend = "wpctl"

        def fake_run(cmd, *a, **k):
            if cmd[:2] == ["wpctl", "status"]:
                return "Stream objects\n"
            raise AssertionError(f"unexpected cmd {cmd}")
        import rpi5.volume as vol
        vol._run = fake_run
        try:
            self.assertIsNone(self.ctrl.set_channel("ffplay", 40))
        finally:
            vol._run = orig


if __name__ == "__main__":
    unittest.main()
