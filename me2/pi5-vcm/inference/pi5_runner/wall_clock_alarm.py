"""Non-persistent, one-shot local wall-clock alarm."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, time, timedelta
from threading import RLock, Timer
from typing import Protocol

logger = logging.getLogger(__name__)


class AlarmHandle(Protocol):
    daemon: bool

    def start(self) -> None: ...

    def cancel(self) -> None: ...


class AlarmScheduler(Protocol):
    def start_alarm(self, hour: int, minute: int) -> datetime: ...

    def close(self) -> None: ...


AlarmFactory = Callable[[float, Callable[[], None]], AlarmHandle]


class WallClockAlarm:
    """Schedule one replaceable alarm at the next local occurrence of a time."""

    def __init__(
        self,
        on_expire: Callable[[datetime], None],
        *,
        now: Callable[[], datetime] = datetime.now,
        alarm_factory: AlarmFactory = Timer,
    ) -> None:
        self._on_expire = on_expire
        self._now = now
        self._alarm_factory = alarm_factory
        self._lock = RLock()
        self._alarm: AlarmHandle | None = None
        self._generation = 0
        self._closed = False

    def start_alarm(self, hour: int, minute: int) -> datetime:
        if (
            isinstance(hour, bool)
            or not isinstance(hour, int)
            or not 0 <= hour <= 23
            or isinstance(minute, bool)
            or not isinstance(minute, int)
            or not 0 <= minute <= 59
        ):
            raise ValueError("alarm time must be a valid local hour and minute")

        now = self._now()
        target = datetime.combine(now.date(), time(hour, minute))
        if target <= now:
            target += timedelta(days=1)
        delay_seconds = (target - now).total_seconds()

        with self._lock:
            if self._closed:
                raise RuntimeError("wall-clock alarm is closed")
            self._generation += 1
            generation = self._generation
            if self._alarm is not None:
                self._alarm.cancel()
            alarm = self._alarm_factory(
                delay_seconds,
                lambda: self._expire(generation, target),
            )
            alarm.daemon = True
            self._alarm = alarm
            alarm.start()
        return target

    def cancel(self) -> bool:
        with self._lock:
            self._generation += 1
            alarm = self._alarm
            self._alarm = None
            if alarm is None:
                return False
            alarm.cancel()
            return True

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self.cancel()

    def _expire(self, generation: int, target: datetime) -> None:
        with self._lock:
            if self._closed or generation != self._generation:
                return
            self._generation += 1
            self._alarm = None
        try:
            self._on_expire(target)
        except Exception:
            logger.exception("wall-clock alarm expiry callback failed")
