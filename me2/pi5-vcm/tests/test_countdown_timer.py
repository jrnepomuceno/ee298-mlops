from __future__ import annotations

from collections.abc import Callable

import pytest

from inference.pi5_runner.countdown_timer import CountdownTimer


class FakeTimer:
    def __init__(self, interval: float, callback: Callable[[], None]) -> None:
        self.interval = interval
        self.callback = callback
        self.daemon = False
        self.started = False
        self.cancelled = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if self.started and not self.cancelled:
            self.callback()


def test_countdown_timer_fires_expiry_callback_once() -> None:
    timers: list[FakeTimer] = []
    expiries: list[str] = []

    def timer_factory(
        interval: float,
        callback: Callable[[], None],
    ) -> FakeTimer:
        timer = FakeTimer(interval, callback)
        timers.append(timer)
        return timer

    countdown = CountdownTimer(
        lambda: expiries.append("expired"),
        timer_factory=timer_factory,
    )
    countdown.start(10)

    timers[0].fire()
    timers[0].fire()

    assert timers[0].interval == 10
    assert timers[0].daemon
    assert expiries == ["expired"]
    countdown.close()


def test_new_countdown_replaces_previous_timer() -> None:
    timers: list[FakeTimer] = []
    expiries: list[str] = []

    def timer_factory(
        interval: float,
        callback: Callable[[], None],
    ) -> FakeTimer:
        timer = FakeTimer(interval, callback)
        timers.append(timer)
        return timer

    countdown = CountdownTimer(
        lambda: expiries.append("expired"),
        timer_factory=timer_factory,
    )
    countdown.start(10)
    countdown.start(30)
    timers[0].fire()
    timers[1].fire()

    assert timers[0].cancelled
    assert expiries == ["expired"]
    countdown.close()


def test_cancel_and_close_stop_active_countdown() -> None:
    timers: list[FakeTimer] = []

    def timer_factory(
        interval: float,
        callback: Callable[[], None],
    ) -> FakeTimer:
        timer = FakeTimer(interval, callback)
        timers.append(timer)
        return timer

    countdown = CountdownTimer(
        lambda: pytest.fail("cancelled timer must not expire"),
        timer_factory=timer_factory,
    )

    assert not countdown.cancel()
    countdown.start(10)
    assert countdown.cancel()
    timers[0].fire()
    countdown.start(30)
    countdown.close()
    timers[1].fire()

    assert timers[0].cancelled
    assert timers[1].cancelled
    with pytest.raises(RuntimeError, match="closed"):
        countdown.start(10)


@pytest.mark.parametrize("seconds", [0, -1, True, 1.5])
def test_countdown_timer_rejects_invalid_duration(seconds) -> None:
    countdown = CountdownTimer(lambda: None)

    with pytest.raises(ValueError, match="positive integer"):
        countdown.start(seconds)

    countdown.close()
