"""Non-persistent asynchronous countdown timer."""

from __future__ import annotations

import logging
from collections.abc import Callable
from threading import RLock, Timer
from typing import Protocol

logger = logging.getLogger(__name__)


class TimerHandle(Protocol):
    daemon: bool

    def start(self) -> None: ...

    def cancel(self) -> None: ...


class TimerScheduler(Protocol):
    def start(self, seconds: int) -> None: ...

    def close(self) -> None: ...


TimerFactory = Callable[[float, Callable[[], None]], TimerHandle]


class CountdownTimer:
    """Manage one replaceable countdown and invoke a callback on expiry."""

    def __init__(
        self,
        on_expire: Callable[[], None],
        *,
        timer_factory: TimerFactory = Timer,
    ) -> None:
        self._on_expire = on_expire
        self._timer_factory = timer_factory
        self._lock = RLock()
        self._timer: TimerHandle | None = None
        self._generation = 0
        self._closed = False

    def start(self, seconds: int) -> None:
        if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds <= 0:
            raise ValueError("timer duration must be a positive integer")

        with self._lock:
            if self._closed:
                raise RuntimeError("countdown timer is closed")
            self._generation += 1
            generation = self._generation
            if self._timer is not None:
                self._timer.cancel()
            timer = self._timer_factory(
                seconds,
                lambda: self._expire(generation),
            )
            timer.daemon = True
            self._timer = timer
            timer.start()

    def cancel(self) -> bool:
        with self._lock:
            self._generation += 1
            timer = self._timer
            self._timer = None
            if timer is None:
                return False
            timer.cancel()
            return True

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self.cancel()

    def _expire(self, generation: int) -> None:
        with self._lock:
            if self._closed or generation != self._generation:
                return
            self._generation += 1
            self._timer = None
        try:
            self._on_expire()
        except Exception:
            logger.exception("countdown timer expiry callback failed")
