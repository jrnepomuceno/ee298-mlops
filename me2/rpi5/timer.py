"""Process-local timer support for the Pi runtime."""
from __future__ import annotations

from dataclasses import dataclass
import threading
from datetime import datetime, timedelta
from typing import Callable
from uuid import uuid4


SECONDS_PER_UNIT = {
    "second": 1.0,
    "minute": 60.0,
    "hour": 3600.0,
}


@dataclass(frozen=True)
class TimerState:
    timer_id: str
    duration: float
    unit: str


@dataclass(frozen=True)
class AlarmState:
    """A wall-clock alarm: rings at a fixed time of day, not a countdown."""

    alarm_id: str
    time: str          # canonical "HH:MM" (24-hour) the alarm is set for
    fires_at: datetime # the concrete datetime it will ring (may be tomorrow)


def parse_alarm_time(time: object) -> tuple[int, int]:
    """Parse an alarm time slot into ``(hour, minute)`` (24-hour).

    Accepts the shapes the slot parser produces: ``"HH:MM"``, ``"HH:MM AM/PM"``,
    or a bare hour ``"7"``. Raises :class:`ValueError` when it cannot be read,
    or when the hour/minute fall outside a valid clock time.
    """
    text = str(time).strip().upper()
    if not text:
        raise ValueError("empty alarm time")

    ampm = ""
    if "AM" in text:
        ampm, text = "AM", text.replace("AM", " ")
    elif "PM" in text:
        ampm, text = "PM", text.replace("PM", " ")
    text = text.strip()

    if ":" in text:
        parts = text.split(":")
        hour_s, minute_s = parts[0], parts[1]
    else:
        hour_s, minute_s = text, "0"

    try:
        hour, minute = int(hour_s), int(minute_s)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unparseable alarm time {time!r}") from exc

    if ampm == "PM" and hour < 12:
        hour += 12
    elif ampm == "AM" and hour == 12:
        hour = 0

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"alarm time out of range: {time!r}")
    return hour, minute


class TimerManager:
    """Maintain one cancellable in-process timer and one wall-clock alarm.

    The two share a single ``on_expire`` callback but are scheduled
    independently: a countdown timer and an alarm can both be armed at once,
    and cancelling one never touches the other.
    """

    def __init__(self, on_expire: Callable[[TimerState], None] | None = None) -> None:
        self.on_expire = on_expire
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._state: TimerState | None = None
        self._alarm: threading.Timer | None = None
        self._alarm_state: AlarmState | None = None

    def start(self, duration: object, unit: object) -> dict[str, object]:
        try:
            numeric_duration = float(duration)
        except (TypeError, ValueError) as exc:
            raise ValueError("timer duration must be numeric") from exc
        normalized_unit = str(unit).rstrip("s").lower()
        if numeric_duration <= 0 or normalized_unit not in SECONDS_PER_UNIT:
            raise ValueError("timer duration or unit is invalid")

        with self._lock:
            self._cancel_locked()
            state = TimerState(
                timer_id=uuid4().hex,
                duration=numeric_duration,
                unit=normalized_unit,
            )
            timer = threading.Timer(
                numeric_duration * SECONDS_PER_UNIT[normalized_unit],
                self._expire,
                args=(state,),
            )
            timer.daemon = True
            self._state = state
            self._timer = timer
            timer.start()
        return {
            "status": "executed",
            "action": "timer.set",
            "timer_id": state.timer_id,
            "duration": state.duration,
            "duration_unit": state.unit,
            "side_effects": True,
        }

    def set_timer(self, duration: object, unit: object = "minute") -> dict[str, object]:
        """Convenience wrapper for :meth:`start` used by the timer executor.

        Accepts ``(duration, unit)`` straight from the model slots, e.g.
        ``set_timer(30, "second")`` or ``set_timer(5, "minute")``. The unit is
        honored (second/minute/hour); it is never silently treated as minutes.
        """
        return self.start(duration, unit)

    def set_alarm(self, time: object, now: datetime | None = None) -> dict[str, object]:
        """Schedule a wall-clock alarm (distinct from a countdown timer).

        ``time`` is a slot value (``"HH:MM"``, ``"HH:MM AM/PM"``, or a bare
        hour). The alarm fires at the next occurrence of that clock time; if it
        is already past today it rolls forward to the next day. Single-shot and
        in-process: it does not persist across a reboot, and arming a new alarm
        replaces any previous one.

        ``now`` is injectable for tests; defaults to the wall clock.
        """
        hour, minute = parse_alarm_time(time)
        base = now or datetime.now()
        fires_at = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if fires_at <= base:
            fires_at += timedelta(days=1)

        delay = (fires_at - base).total_seconds()
        with self._lock:
            self._clear_alarm_locked()
            state = AlarmState(
                alarm_id=uuid4().hex,
                time=f"{hour:02d}:{minute:02d}",
                fires_at=fires_at,
            )
            timer = threading.Timer(delay, self._alarm_expire, args=(state,))
            timer.daemon = True
            self._alarm_state = state
            self._alarm = timer
            timer.start()
        return {
            "status": "executed",
            "action": "alarm.set",
            "alarm_id": state.alarm_id,
            "time": state.time,
            "fires_at": state.fires_at.isoformat(),
            "delay_seconds": delay,
            "side_effects": True,
        }

    def cancel(self) -> dict[str, object]:
        """Cancel the active countdown timer (does not touch the alarm)."""
        with self._lock:
            state = self._state
            self._cancel_locked()
        return {
            "status": "executed" if state else "rejected",
            "action": "timer.cancel" if state else None,
            "timer_id": state.timer_id if state else None,
            "reason": None if state else "no_active_timer",
            "side_effects": bool(state),
        }

    def cancel_alarm(self) -> dict[str, object]:
        """Cancel the armed alarm (does not touch the countdown timer)."""
        with self._lock:
            state = self._alarm_state
            self._clear_alarm_locked()
        return {
            "status": "executed" if state else "rejected",
            "action": "alarm.cancel" if state else None,
            "alarm_id": state.alarm_id if state else None,
            "reason": None if state else "no_active_alarm",
            "side_effects": bool(state),
        }

    def close(self) -> None:
        with self._lock:
            self._cancel_locked()
            self._clear_alarm_locked()

    def _expire(self, state: TimerState) -> None:
        with self._lock:
            if self._state != state:
                return
            self._timer = None
            self._state = None
        if self.on_expire is not None:
            self.on_expire(state)

    def _alarm_expire(self, state: AlarmState) -> None:
        with self._lock:
            if self._alarm_state != state:
                return
            self._alarm = None
            self._alarm_state = None
        if self.on_expire is not None:
            self.on_expire(state)

    def _cancel_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = None
        self._state = None

    def _clear_alarm_locked(self) -> None:
        if self._alarm is not None:
            self._alarm.cancel()
        self._alarm = None
        self._alarm_state = None
