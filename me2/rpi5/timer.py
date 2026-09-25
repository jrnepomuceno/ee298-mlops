"""Process-local timer support for the Pi runtime."""
from __future__ import annotations

from dataclasses import dataclass
import threading
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


class TimerManager:
    """Maintain one cancellable in-process timer."""

    def __init__(self, on_expire: Callable[[TimerState], None] | None = None) -> None:
        self.on_expire = on_expire
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._state: TimerState | None = None

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

    def cancel(self) -> dict[str, object]:
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

    def close(self) -> None:
        with self._lock:
            self._cancel_locked()

    def _expire(self, state: TimerState) -> None:
        with self._lock:
            if self._state != state:
                return
            self._timer = None
            self._state = None
        if self.on_expire is not None:
            self.on_expire(state)

    def _cancel_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = None
        self._state = None
