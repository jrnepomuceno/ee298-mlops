"""Unit tests for the live timer path (P1+P2+P3).

Covers: unit correctness (second/minute/hour), the single-source-of-truth
spoken reply, cancellation, the no-manager fallback, expiry firing the
callback, and the honest "not wired" alarm stub.
"""
from __future__ import annotations

import threading
import time
import unittest

from rpi5.facade import decode
from rpi5.timer import TimerManager
from rpi5.executors.timer import TimerExecutor, _spoken_set
from rpi5.orchestrator import default_orchestrator


class FakeTimer:
    """Records the seconds handed to the underlying scheduling call."""

    def __init__(self, interval, func, args=()):
        self.interval = interval
        self.func = func
        self.args = args
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class RecordingManager:
    """Stand-in for TimerManager that captures set_timer/cancel calls."""

    def __init__(self):
        self.set_calls = []
        self.cancel_calls = 0
        self.last = None

    def set_timer(self, duration, unit="minute"):
        from rpi5.timer import SECONDS_PER_UNIT
        u = str(unit).rstrip("s").lower()
        self.set_calls.append((int(duration), u))
        self.last = FakeTimer(int(duration) * SECONDS_PER_UNIT[u], lambda s: None)
        return {"status": "executed", "action": "timer.set", "side_effects": True}

    def cancel(self):
        self.cancel_calls += 1
        had = self.last is not None
        return {"status": "executed" if had else "rejected",
                "side_effects": bool(had)}


class SpokenSetFormatTests(unittest.TestCase):
    def test_seconds_plural(self):
        self.assertEqual(_spoken_set(30, "second"), "Timer set for 30 seconds.")

    def test_minutes_plural(self):
        self.assertEqual(_spoken_set(5, "minute"), "Timer set for 5 minutes.")

    def test_singular_minute(self):
        self.assertEqual(_spoken_set(1, "minute"), "Timer set for 1 minute.")

    def test_singular_hour(self):
        self.assertEqual(_spoken_set(1, "hour"), "Timer set for 1 hour.")

    def test_trailing_s_stripped(self):
        self.assertEqual(_spoken_set(30, "seconds"), "Timer set for 30 seconds.")


class TimerUnitCorrectnessTests(unittest.TestCase):
    """Regression: the unit must be honored, never silently treated as minutes."""

    def _exec(self):
        mgr = RecordingManager()
        return mgr, TimerExecutor(dry_run=False, manager=mgr)

    def test_seconds_not_treated_as_minutes(self):
        mgr, ex = self._exec()
        req = decode("set_timer", {"duration": 30, "duration_unit": "second"}, 0.99,
                     dry_run=False)
        res = ex.run(req)
        self.assertTrue(res.ok)
        self.assertTrue(res.side_effects)
        # 30 seconds -> 30s scheduled, NOT 1800s.
        self.assertEqual(mgr.last.interval, 30.0)

    def test_minutes_scheduled_correctly(self):
        mgr, ex = self._exec()
        req = decode("set_timer", {"duration": 5, "duration_unit": "minute"}, 0.99,
                     dry_run=False)
        ex.run(req)
        self.assertEqual(mgr.last.interval, 300.0)

    def test_hours_scheduled_correctly(self):
        mgr, ex = self._exec()
        req = decode("set_timer", {"duration": 2, "duration_unit": "hour"}, 0.99,
                     dry_run=False)
        ex.run(req)
        self.assertEqual(mgr.last.interval, 7200.0)

    def test_default_unit_is_minute_when_absent(self):
        mgr, ex = self._exec()
        req = decode("set_timer", {"duration": 4}, 0.99, dry_run=False)
        ex.run(req)
        self.assertEqual(mgr.set_calls[-1], (4, "minute"))
        self.assertEqual(mgr.last.interval, 240.0)


class TimerReplySourceTests(unittest.TestCase):
    """The executor's payload['answer'] is the single source of truth."""

    def test_spoken_reply_equals_executor_answer(self):
        mgr = RecordingManager()
        orch = default_orchestrator(dry_run=False, timer_manager=mgr)
        req = decode("set_timer", {"duration": 30, "duration_unit": "second"}, 0.99,
                     dry_run=False)
        result = orch.run(req)
        self.assertTrue(result.handled)
        self.assertEqual(result.reply_text, "Timer set for 30 seconds.")
        self.assertEqual(result.execution.payload.get("answer"),
                         "Timer set for 30 seconds.")

    def test_divergent_live_answer_overrides_template(self):
        # A live answer differing from the facade template must win.
        mgr = RecordingManager()
        orch = default_orchestrator(dry_run=False, timer_manager=mgr)
        req = decode("set_timer", {"duration": 1, "duration_unit": "hour"}, 0.99,
                     dry_run=False)
        result = orch.run(req)
        self.assertEqual(result.reply_text, "Timer set for 1 hour.")


class TimerCancelTests(unittest.TestCase):
    def test_cancel_reports_stopped_when_active(self):
        mgr = RecordingManager()
        mgr.set_timer(30, "second")  # ensure one is active
        ex = TimerExecutor(dry_run=False, manager=mgr)
        req = decode("stop_timer", {}, 0.99, dry_run=False)
        res = ex.run(req)
        self.assertTrue(res.ok)
        self.assertEqual(res.payload.get("answer"), "Timer stopped.")
        self.assertEqual(mgr.cancel_calls, 1)

    def test_cancel_reports_none_running(self):
        mgr = RecordingManager()
        ex = TimerExecutor(dry_run=False, manager=mgr)
        req = decode("stop_timer", {}, 0.99, dry_run=False)
        res = ex.run(req)
        self.assertTrue(res.ok)
        self.assertEqual(res.payload.get("answer"), "There's no timer running.")


class TimerFallbackTests(unittest.TestCase):
    def test_no_manager_fails_soft(self):
        ex = TimerExecutor(dry_run=False, manager=None)
        req = decode("set_timer", {"duration": 30, "duration_unit": "second"}, 0.99,
                     dry_run=False)
        res = ex.run(req)
        self.assertFalse(res.ok)
        self.assertIn("no TimerManager", res.detail)
        self.assertFalse(res.side_effects)

    def test_dry_run_has_no_side_effects(self):
        ex = TimerExecutor(dry_run=True, manager=RecordingManager())
        req = decode("set_timer", {"duration": 30, "duration_unit": "second"}, 0.99,
                     dry_run=True)
        res = ex.run(req)
        self.assertTrue(res.ok)
        self.assertFalse(res.side_effects)


class AlarmStubTests(unittest.TestCase):
    def test_alarm_reports_not_wired(self):
        mgr = RecordingManager()
        ex = TimerExecutor(dry_run=False, manager=mgr)
        req = decode("set_alarm", {"time": "07:00"}, 0.99, dry_run=False)
        res = ex.run(req)
        # Graceful ack (like the weather fallback): recognized, no side effect,
        # and the honest reason is what gets spoken.
        self.assertTrue(res.ok)
        self.assertIn("not wired", res.detail)
        self.assertFalse(res.side_effects)
        self.assertEqual(res.payload.get("answer"), "Alarms aren't available yet.")

    def test_alarm_spoken_via_orchestrator(self):
        mgr = RecordingManager()
        orch = default_orchestrator(dry_run=False, timer_manager=mgr)
        req = decode("set_alarm", {"time": "07:00"}, 0.99, dry_run=False)
        result = orch.run(req)
        self.assertEqual(result.reply_text, "Alarms aren't available yet.")

    def test_manager_set_alarm_raises_not_implemented(self):
        mgr = TimerManager()
        with self.assertRaises(NotImplementedError):
            mgr.set_alarm("07:00")


class TimerExpiryTests(unittest.TestCase):
    """The real TimerManager must fire on_expire when the countdown elapses."""

    def test_expiry_fires_callback(self):
        fired = threading.Event()
        seen = {}

        def cb(state):
            seen["state"] = state
            fired.set()

        mgr = TimerManager(on_expire=cb)
        mgr.set_timer(1, "second")  # 1 second
        self.assertTrue(fired.wait(timeout=5.0), "on_expire did not fire")
        self.assertEqual(seen["state"].duration, 1.0)
        self.assertEqual(seen["state"].unit, "second")
        mgr.close()

    def test_cancel_prevents_expiry(self):
        fired = threading.Event()
        mgr = TimerManager(on_expire=lambda s: fired.set())
        mgr.set_timer(2, "second")
        mgr.cancel()
        time.sleep(2.2)
        self.assertFalse(fired.is_set(), "cancelled timer should not fire")
        mgr.close()


if __name__ == "__main__":
    unittest.main()
