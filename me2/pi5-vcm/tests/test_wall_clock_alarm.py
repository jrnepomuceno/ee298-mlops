from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import pytest

from inference.pi5_runner.wall_clock_alarm import WallClockAlarm


class FakeAlarm:
    def __init__(self, delay: float, callback: Callable[[], None]) -> None:
        self.delay = delay
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


def create_alarm(
    now: datetime,
    alarms: list[FakeAlarm],
    expiries: list[datetime],
) -> WallClockAlarm:
    def alarm_factory(delay: float, callback: Callable[[], None]) -> FakeAlarm:
        alarm = FakeAlarm(delay, callback)
        alarms.append(alarm)
        return alarm

    return WallClockAlarm(
        expiries.append,
        now=lambda: now,
        alarm_factory=alarm_factory,
    )


def test_alarm_schedules_next_local_occurrence_today() -> None:
    now = datetime(2026, 10, 20, 5, 0)
    alarms: list[FakeAlarm] = []
    expiries: list[datetime] = []
    alarm = create_alarm(now, alarms, expiries)

    target = alarm.start_alarm(6, 0)
    alarms[0].fire()

    assert target == datetime(2026, 10, 20, 6, 0)
    assert alarms[0].delay == pytest.approx(3600)
    assert alarms[0].daemon
    assert expiries == [target]
    alarm.close()


def test_alarm_rolls_forward_when_time_has_passed() -> None:
    now = datetime(2026, 10, 20, 8, 1)
    alarms: list[FakeAlarm] = []
    expiries: list[datetime] = []
    alarm = create_alarm(now, alarms, expiries)

    target = alarm.start_alarm(8, 0)

    assert target == datetime(2026, 10, 21, 8, 0)
    assert alarms[0].delay == pytest.approx(23 * 60 * 60 + 59 * 60)
    alarm.close()


def test_new_alarm_replaces_previous_and_close_cancels() -> None:
    now = datetime(2026, 10, 20, 5, 0)
    alarms: list[FakeAlarm] = []
    expiries: list[datetime] = []
    alarm = create_alarm(now, alarms, expiries)

    alarm.start_alarm(6, 0)
    alarm.start_alarm(8, 0)
    alarms[0].fire()
    alarm.close()
    alarms[1].fire()

    assert alarms[0].cancelled
    assert alarms[1].cancelled
    assert expiries == []
    with pytest.raises(RuntimeError, match="closed"):
        alarm.start_alarm(9, 0)


@pytest.mark.parametrize(
    ("hour", "minute"),
    [(24, 0), (-1, 0), (6, 60), (6, -1), (True, 0), (6, True)],
)
def test_alarm_rejects_invalid_clock_values(hour: int, minute: int) -> None:
    alarm = WallClockAlarm(lambda target: None)

    with pytest.raises(ValueError, match="valid local hour and minute"):
        alarm.start_alarm(hour, minute)

    alarm.close()


def test_expiry_callback_failure_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    now = datetime(2026, 10, 20, 5, 0)
    alarms: list[FakeAlarm] = []

    def alarm_factory(delay: float, callback: Callable[[], None]) -> FakeAlarm:
        alarm = FakeAlarm(delay, callback)
        alarms.append(alarm)
        return alarm

    alarm = WallClockAlarm(
        lambda target: (_ for _ in ()).throw(RuntimeError("speech failed")),
        now=lambda: now,
        alarm_factory=alarm_factory,
    )

    alarm.start_alarm(6, 0)
    alarms[0].fire()

    assert "wall-clock alarm expiry callback failed" in caplog.text
    alarm.close()
