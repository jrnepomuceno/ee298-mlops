"""Unit tests for the live timer path (P1+P2+P3).

Covers: unit correctness (second/minute/hour), the single-source-of-truth
spoken reply, cancellation, the no-manager fallback, expiry firing the
callback, and the honest "not wired" alarm stub.
"""
from __future__ import annotations

import threading
import time
import unittest
from datetime import timedelta

from rpi5.facade import decode
from rpi5.timer import TimerAlarm, TimerManager
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

    def set_alarm(self, time, now=None):
        from rpi5.timer import parse_alarm_time
        h, m = parse_alarm_time(time)
        self.alarm_time = f"{h:02d}:{m:02d}"
        self.alarm_calls = getattr(self, "alarm_calls", 0) + 1
        return {"status": "executed", "action": "alarm.set",
                "time": self.alarm_time, "side_effects": True}


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


class AlarmSetTests(unittest.TestCase):
    """Wall-clock alarms: the executor schedules and confirms the time."""

    def test_alarm_sets_and_confirms(self):
        mgr = RecordingManager()
        ex = TimerExecutor(dry_run=False, manager=mgr)
        req = decode("set_alarm", {"time": "07:00"}, 0.99, dry_run=False)
        res = ex.run(req)
        self.assertTrue(res.ok)
        self.assertTrue(res.side_effects)
        self.assertEqual(mgr.alarm_time, "07:00")
        self.assertEqual(res.payload.get("answer"), "Alarm set for 07:00.")

    def test_alarm_ampm_normalized(self):
        mgr = RecordingManager()
        ex = TimerExecutor(dry_run=False, manager=mgr)
        req = decode("set_alarm", {"time": "7:00 PM"}, 0.99, dry_run=False)
        res = ex.run(req)
        self.assertTrue(res.ok)
        self.assertEqual(mgr.alarm_time, "19:00")
        self.assertEqual(res.payload.get("answer"), "Alarm set for 19:00.")

    def test_alarm_invalid_time_fails_soft(self):
        mgr = RecordingManager()
        ex = TimerExecutor(dry_run=False, manager=mgr)
        req = decode("set_alarm", {"time": "25:00"}, 0.99, dry_run=False)
        res = ex.run(req)
        self.assertFalse(res.ok)
        self.assertIn("invalid alarm time", res.detail)
        self.assertFalse(res.side_effects)

    def test_alarm_no_manager_fails(self):
        ex = TimerExecutor(dry_run=False, manager=None)
        req = decode("set_alarm", {"time": "07:00"}, 0.99, dry_run=False)
        res = ex.run(req)
        self.assertFalse(res.ok)
        self.assertIn("no TimerManager", res.detail)


class AlarmManagerTests(unittest.TestCase):
    """The real TimerManager schedules a wall-clock alarm and rolls forward."""

    def test_parse_alarm_time_forms(self):
        from rpi5.timer import parse_alarm_time
        self.assertEqual(parse_alarm_time("07:00"), (7, 0))
        self.assertEqual(parse_alarm_time("7:00 PM"), (19, 0))
        self.assertEqual(parse_alarm_time("12:00 AM"), (0, 0))
        self.assertEqual(parse_alarm_time("12:30 PM"), (12, 30))
        self.assertEqual(parse_alarm_time("9"), (9, 0))
        for bad in ("", "25:00", "12:60", "abc"):
            with self.assertRaises(ValueError):
                parse_alarm_time(bad)

    def test_alarm_schedules_at_next_occurrence(self):
        from datetime import datetime
        mgr = TimerManager()
        # 1 minute from now, expressed as a wall-clock time (alarms are
        # minute-resolution, so a 1-minute lead gives a 60s delay).
        now = datetime(2026, 9, 28, 12, 0, 0)
        target = (now + timedelta(minutes=1)).strftime("%H:%M")
        res = mgr.set_alarm(target, now=now)
        self.assertEqual(res["delay_seconds"], 60.0)
        self.assertEqual(res["time"], "12:01")
        self.assertEqual(res["fires_at"], "2026-09-28T12:01:00")
        self.assertIsNotNone(mgr._alarm_state)
        mgr.close()

    def test_alarm_rolls_forward_when_past(self):
        from datetime import datetime
        mgr = TimerManager()
        now = datetime(2026, 9, 28, 22, 0, 0)  # 10pm
        res = mgr.set_alarm("07:00", now=now)  # 7am already passed today
        self.assertEqual(res["time"], "07:00")
        # Should fire tomorrow morning (~9 hours away), not immediately.
        self.assertGreater(res["delay_seconds"], 8 * 3600)
        self.assertLess(res["delay_seconds"], 10 * 3600)
        mgr.close()

    def test_alarm_independent_of_timer(self):
        from datetime import datetime
        mgr = TimerManager()
        now = datetime(2026, 9, 28, 12, 0, 0)
        mgr.set_alarm("12:01", now=now)
        # Cancelling the countdown timer must not clear the alarm.
        mgr.cancel()
        self.assertIsNotNone(mgr._alarm_state)
        mgr.close()

    def test_cancel_alarm_clears_it(self):
        from datetime import datetime
        mgr = TimerManager()
        now = datetime(2026, 9, 28, 12, 0, 0)
        mgr.set_alarm("12:01", now=now)
        res = mgr.cancel_alarm()
        self.assertEqual(res["status"], "executed")
        self.assertIsNone(mgr._alarm_state)
        mgr.close()


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


class TimerAlarmLoopTests(unittest.TestCase):
    def test_announces_immediately_and_repeats_until_stopped(self):
        announcements = []
        rings = []
        announced_three = threading.Event()

        def announce():
            announcements.append("Your timer is up.")
            if len(announcements) >= 3:
                announced_three.set()

        alarm = TimerAlarm(lambda: rings.append("ring"), announce,
                           announce_interval_s=0.03, ring_gap_s=0.001)
        self.assertTrue(alarm.start())
        self.assertTrue(announced_three.wait(timeout=1.0))
        self.assertGreaterEqual(len(rings), 1)
        self.assertTrue(alarm.stop())
        self.assertFalse(alarm.active)

    def test_pause_stops_ring_until_resumed(self):
        rings = []
        alarm = TimerAlarm(lambda: rings.append("ring"), lambda: None,
                           announce_interval_s=10, ring_gap_s=0.005)
        alarm.start()
        self.assertTrue(alarm.wait_first_ring(timeout=1.0))
        self.assertTrue(alarm.pause())
        paused_count = len(rings)
        time.sleep(0.04)
        self.assertEqual(len(rings), paused_count)
        self.assertTrue(alarm.resume())
        deadline = time.monotonic() + 1.0
        while len(rings) == paused_count and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertGreater(len(rings), paused_count)
        alarm.stop()

    def test_stop_is_idempotent(self):
        alarm = TimerAlarm(lambda: None, lambda: None)
        self.assertFalse(alarm.stop())
        self.assertTrue(alarm.start())
        self.assertTrue(alarm.stop())
        self.assertFalse(alarm.stop())


class TimerExecutorAlarmTests(unittest.TestCase):
    def test_stop_timer_stops_ringing_after_countdown_has_expired(self):
        from rpi5.timer import TimerAlarm

        alarm = TimerAlarm(lambda: None, lambda: None, ring_gap_s=0.01)
        alarm.start()
        self.assertTrue(alarm.wait_first_ring(timeout=1.0))
        executor = TimerExecutor(dry_run=False, manager=TimerManager(), alarm=alarm)
        request = decode("stop_timer", {}, 0.99, dry_run=False)

        result = executor.run(request)

        self.assertTrue(result.ok)
        self.assertTrue(result.side_effects)
        self.assertEqual(result.payload["answer"], "Timer stopped.")
        self.assertFalse(alarm.active)

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
