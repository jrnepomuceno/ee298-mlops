"""Unit tests for the live HVAC path (audio-only temperature set-point).

Two layers are exercised:

* The **facade** already validates the ``temperature`` slot (10-35) and
  rejects out-of-band / missing / non-numeric values as ``invalid_slot``
  before any executor runs. Those cases are asserted at the facade boundary.
* The :class:`HvacExecutor` confirms a valid set-point by voice, records the
  requested value in a structured payload a future controller can consume,
  and fails soft on an unknown action code. Nothing physical is touched.
"""
from __future__ import annotations

import unittest

from rpi5.facade import decode, RejectResult
from rpi5.executors.hvac import HvacExecutor, TEMP_MIN, TEMP_MAX
from rpi5.orchestrator import default_orchestrator


def _decode(intent: str, slots: dict | None = None, confidence: float = 0.95):
    return decode(intent, slots or {}, confidence)


def _req(intent: str, slots: dict | None = None, confidence: float = 0.95):
    """Decode and assert the intent was *accepted* (returns ActionRequest)."""
    decoded = _decode(intent, slots, confidence)
    assert not isinstance(decoded, RejectResult), f"{intent} unexpectedly rejected"
    return decoded


class TestFacadeValidatesTemperature(unittest.TestCase):
    """The gate rejects bad set-points before the executor ever runs."""

    def test_in_band_accepted(self):
        self.assertNotIsInstance(_decode("set_temperature", {"temperature": 24}), RejectResult)

    def test_out_of_band_rejected(self):
        for temp in (TEMP_MIN - 1, TEMP_MAX + 1, 0, 99):
            self.assertIsInstance(_decode("set_temperature", {"temperature": temp}),
                                  RejectResult, f"{temp} should be rejected")

    def test_missing_slot_rejected(self):
        self.assertIsInstance(_decode("set_temperature", {}), RejectResult)

    def test_non_numeric_rejected(self):
        self.assertIsInstance(_decode("set_temperature", {"temperature": "warm"}), RejectResult)


class TestHvacSetTemperature(unittest.TestCase):
    def test_valid_setpoint_confirms_by_voice(self):
        ex = HvacExecutor(dry_run=False)
        res = ex.run(_req("set_temperature", {"temperature": 24}))
        self.assertTrue(res.ok)
        self.assertFalse(res.side_effects)          # audio-only, no hardware
        self.assertEqual(res.payload["temperature"], 24)
        self.assertEqual(res.payload["unit"], "C")
        self.assertEqual(res.payload["answer"], "Setting the temperature to 24 degrees.")

    def test_band_edges_accepted(self):
        ex = HvacExecutor(dry_run=False)
        for temp in (TEMP_MIN, TEMP_MAX):
            res = ex.run(_req("set_temperature", {"temperature": temp}))
            self.assertTrue(res.ok, f"edge {temp} should be accepted")
            self.assertEqual(res.payload["temperature"], temp)

    def test_unsupported_action_rejected(self):
        # thermostat.set is the only hvac action; anything else fails soft.
        ex = HvacExecutor(dry_run=False)
        req = _req("set_temperature", {"temperature": 24})
        fake = type(req)(**{**req.__dict__, "action_code": "thermostat.blast"})
        res = ex.run(fake)
        self.assertFalse(res.ok)
        self.assertIn("unsupported hvac action", res.detail)

    def test_defensive_bad_slot_rejected(self):
        # Bypass the facade gate to prove the executor guards on its own.
        ex = HvacExecutor(dry_run=False)
        req = _req("set_temperature", {"temperature": 24})
        for bad in ({}, {"temperature": "warm"}, {"temperature": 99}):
            fake = type(req)(**{**req.__dict__, "slots": bad})
            res = ex.run(fake)
            self.assertFalse(res.ok, f"slot {bad!r} should be rejected")

    def test_dry_run_no_side_effects(self):
        ex = HvacExecutor(dry_run=True)
        res = ex.run(_req("set_temperature", {"temperature": 24}))
        self.assertTrue(res.ok)
        self.assertFalse(res.side_effects)
        self.assertNotIn("answer", res.payload)


class TestHvacWiring(unittest.TestCase):
    def test_default_orchestrator_registers_hvac(self):
        orch = default_orchestrator(dry_run=False, speak=lambda *_: None)
        self.assertIsInstance(orch._executors.get("hvac"), HvacExecutor)

    def test_end_to_end_through_pipeline(self):
        spoken = []
        orch = default_orchestrator(dry_run=False, speak=lambda text, *_: spoken.append(text))
        result = orch.run(_req("set_temperature", {"temperature": 21}), source="microphone")
        self.assertTrue(result.handled)
        self.assertIsNotNone(result.execution)
        self.assertTrue(result.execution.ok)
        # The spoken confirmation is what the user hears.
        self.assertIn("Setting the temperature to 21 degrees.", spoken)


if __name__ == "__main__":
    unittest.main()
