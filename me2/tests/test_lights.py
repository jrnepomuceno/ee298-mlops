"""Unit tests for the live lights path (HyperX DuoCast ring light).

Covers: the pluggable :class:`LightDriver` registry, the
:class:`HyperxDuoCastDriver` brightness->colour mapping and its degrade-when-
absent behaviour, the :class:`LightExecutor` on/off/dim mapping (including the
no-driver and unsupported-action failures), and the dry-run fallback. No real
hardware is touched -- the driver's ``subprocess.run`` is patched to record
the command line, exactly like the media tests fake the player.
"""
from __future__ import annotations

import unittest
from unittest import mock

from rpi5.facade import decode
from rpi5.rgb import (HyperxDuoCastDriver, LightDriver, LIGHT_DRIVERS,
                      make_light_driver)
from rpi5.executors.light import LightExecutor
from rpi5.orchestrator import default_orchestrator


def _req(intent: str, slots: dict | None = None, confidence: float = 0.95):
    """Decode an intent the same way the pipeline does, returning the
    :class:`ActionRequest` (asserts it was accepted, not rejected)."""
    decoded = decode(intent, slots or {}, confidence)
    assert not hasattr(decoded, "reason"), f"{intent} unexpectedly rejected"
    return decoded


class FakeRun:
    """Records ``subprocess.run`` command lines; raises on demand."""

    def __init__(self, fail: bool = False):
        self.calls = []
        self.fail = fail

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        if self.fail:
            raise OSError("boom")
        return mock.Mock(returncode=0)


class TestLightDriverRegistry(unittest.TestCase):
    def test_hyperx_registered(self):
        self.assertIn("hyperx-duocast", LIGHT_DRIVERS)
        self.assertIs(LIGHT_DRIVERS["hyperx-duocast"], HyperxDuoCastDriver)

    def test_make_returns_none_for_empty(self):
        self.assertIsNone(make_light_driver(None))
        self.assertIsNone(make_light_driver(""))

    def test_make_returns_named_driver(self):
        drv = make_light_driver("hyperx-duocast", executable="quadcastrgb")
        self.assertIsInstance(drv, HyperxDuoCastDriver)
        self.assertEqual(drv.executable, "quadcastrgb")

    def test_make_unknown_falls_back_to_none(self):
        with mock.patch("sys.stderr") as err:
            self.assertIsNone(make_light_driver("does-not-exist"))
        self.assertTrue(err.write.called)

    def test_is_abstract(self):
        with self.assertRaises(TypeError):
            LightDriver()  # abstract: on/off/set_brightness not implemented


class TestHyperxDuoCastDriver(unittest.TestCase):
    def setUp(self):
        self.drv = HyperxDuoCastDriver(executable="/opt/bin/quadcastrgb")
        self.fake = FakeRun()
        self.run_patcher = mock.patch("rpi5.rgb.subprocess.run", self.fake)
        self.run_patcher.start()
        self.addCleanup(self.run_patcher.stop)
        # Force the driver to believe the device is present so the command
        # actually reaches the (fake) subprocess instead of degrading.
        self.avail_patcher = mock.patch.object(
            self.drv, "available", return_value=True)
        self.avail_patcher.start()
        self.addCleanup(self.avail_patcher.stop)

    def test_available_when_absolute_path_exists(self):
        # Fresh driver: setUp forces self.drv.available() to True, so probe
        # the real availability logic on a separate instance.
        d = HyperxDuoCastDriver(executable="/opt/bin/quadcastrgb")
        with mock.patch("os.path.exists", return_value=True):
            self.assertTrue(d.available())
        with mock.patch("os.path.exists", return_value=False):
            self.assertFalse(d.available())

    def test_on_restores_last_brightness(self):
        self.drv.set_brightness(40)
        self.drv.off()
        self.drv.on()  # should restore 40
        last = self.fake.calls[-1]
        self.assertEqual(last[0], "/opt/bin/quadcastrgb")
        self.assertEqual(last[1], "solid")
        # 40% of FFF4E6 -> FF*0.4=66, F4*0.4=62, E6*0.4=5C
        self.assertEqual(last[2], "66625C")

    def test_off_sends_black(self):
        self.drv.off()
        self.assertEqual(self.fake.calls[-1][2], "000000")

    def test_set_brightness_clamps_and_returns(self):
        self.assertEqual(self.drv.set_brightness(150), 100)
        self.assertEqual(self.drv.set_brightness(0), 1)
        self.assertEqual(self.drv.set_brightness(50), 50)

    def test_full_brightness_is_lit_colour(self):
        self.drv.set_brightness(100)
        self.assertEqual(self.fake.calls[-1][2], "FFF4E6")

    def test_degrades_quietly_when_absent(self):
        drv = HyperxDuoCastDriver(executable="no-such-binary")
        with mock.patch("shutil.which", return_value=None):
            self.assertFalse(drv.available())
            # No exception, and no subprocess call is attempted.
            with mock.patch("rpi5.rgb.subprocess.run") as run:
                drv.set_brightness(50)
                run.assert_not_called()

    def test_close_turns_off(self):
        self.drv.close()
        self.assertEqual(self.fake.calls[-1][2], "000000")


class TestLightExecutor(unittest.TestCase):
    def _executor(self, driver, dry_run=True):
        return LightExecutor(dry_run=dry_run, driver=driver)

    def test_no_driver_live_reports_failure(self):
        ex = self._executor(None, dry_run=False)
        res = ex.run(_req("turn_on_lights"))
        self.assertFalse(res.ok)
        self.assertIn("no light driver", res.detail)
        self.assertFalse(res.side_effects)

    def test_dry_run_accepts_without_driver(self):
        ex = self._executor(None, dry_run=True)
        res = ex.run(_req("dim_lights", {"percent": 50}))
        self.assertTrue(res.ok)
        self.assertFalse(res.side_effects)
        self.assertIn("dry-run", res.detail)

    def test_live_on(self):
        drv = HyperxDuoCastDriver(executable="/opt/bin/quadcastrgb")
        with mock.patch.object(drv, "available", return_value=True), \
             mock.patch("rpi5.rgb.subprocess.run") as run:
            ex = self._executor(drv, dry_run=False)
            res = ex.run(_req("turn_on_lights"))
        self.assertTrue(res.ok)
        self.assertTrue(res.side_effects)
        self.assertEqual(res.payload["driver"], "hyperx-duocast")
        run.assert_called_once()

    def test_live_off(self):
        drv = HyperxDuoCastDriver(executable="/opt/bin/quadcastrgb")
        with mock.patch.object(drv, "available", return_value=True), \
             mock.patch("rpi5.rgb.subprocess.run") as run:
            ex = self._executor(drv, dry_run=False)
            res = ex.run(_req("turn_off_lights"))
        self.assertTrue(res.ok)
        self.assertEqual(res.payload["level"], 0)
        self.assertEqual(run.call_args[0][0][2], "000000")

    def test_live_dim_reports_applied_level(self):
        drv = HyperxDuoCastDriver(executable="/opt/bin/quadcastrgb")
        with mock.patch.object(drv, "available", return_value=True), \
             mock.patch("rpi5.rgb.subprocess.run") as run:
            ex = self._executor(drv, dry_run=False)
            res = ex.run(_req("dim_lights", {"percent": 30}))
        self.assertTrue(res.ok)
        self.assertEqual(res.payload["level"], 30)
        self.assertIn("30", res.detail)

    def test_live_dim_clamps_out_of_range(self):
        # The facade rejects percent>100, so clamp is a defence-in-depth check
        # exercised directly against the executor.
        drv = HyperxDuoCastDriver(executable="/opt/bin/quadcastrgb")
        req = _req("dim_lights", {"percent": 100})
        with mock.patch.object(drv, "available", return_value=True), \
             mock.patch("rpi5.rgb.subprocess.run"):
            ex = self._executor(drv, dry_run=False)
            res = ex.run(req)
        self.assertTrue(res.ok)
        self.assertEqual(res.payload["level"], 100)

    def test_live_failure_is_caught(self):
        drv = HyperxDuoCastDriver(executable="/opt/bin/quadcastrgb")
        with mock.patch.object(drv, "available", return_value=True), \
             mock.patch("rpi5.rgb.subprocess.run", side_effect=OSError("boom")):
            ex = self._executor(drv, dry_run=False)
            res = ex.run(_req("turn_on_lights"))
        self.assertFalse(res.ok)
        self.assertIn("light control failed", res.detail)
        self.assertFalse(res.side_effects)


class TestLightOrchestration(unittest.TestCase):
    def test_default_orchestrator_routes_lights_to_driver(self):
        drv = HyperxDuoCastDriver(executable="/opt/bin/quadcastrgb")
        orch = default_orchestrator(dry_run=False, light_driver=drv)
        with mock.patch.object(drv, "available", return_value=True), \
             mock.patch("rpi5.rgb.subprocess.run") as run:
            result = orch.run(_req("dim_lights", {"percent": 25}),
                              source="test")
        self.assertTrue(result.handled)
        self.assertEqual(result.execution.payload["driver"], "hyperx-duocast")
        self.assertEqual(result.execution.payload["level"], 25)
        run.assert_called_once()

    def test_no_driver_keeps_lights_dry_run(self):
        orch = default_orchestrator(dry_run=True)
        result = orch.run(_req("turn_on_lights"), source="test")
        self.assertTrue(result.handled)
        self.assertFalse(result.execution.side_effects)


if __name__ == "__main__":
    unittest.main()
