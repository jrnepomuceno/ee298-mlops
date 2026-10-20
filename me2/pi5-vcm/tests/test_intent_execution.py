from __future__ import annotations

import signal
from datetime import datetime
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request

import numpy as np
import pytest

from inference.pi5_runner import intent_execution as execution


class FakeCountdownTimer:
    def __init__(self) -> None:
        self.durations: list[int] = []
        self.closed = False

    def start(self, seconds: int) -> None:
        self.durations.append(seconds)

    def close(self) -> None:
        self.closed = True


class FakeAlarmScheduler:
    def __init__(self) -> None:
        self.alarms: list[tuple[int, int]] = []
        self.closed = False

    def start_alarm(self, hour: int, minute: int):
        self.alarms.append((hour, minute))

    def close(self) -> None:
        self.closed = True


class FakeInteractionColorStore:
    def __init__(self, color: str = "Blue") -> None:
        self.color = color

    def load(self) -> str:
        return self.color

    def save(self, color: str) -> None:
        self.color = color


def test_dispatcher_covers_all_model_intents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    monkeypatch.setattr(
        execution.LocalMusicPlayer,
        "play",
        lambda self: execution.ExecutionResult(
            "PLAY_MUSIC", execution.ExecutionStatus.COMPLETED, "test action"
        ),
    )
    monkeypatch.setattr(
        execution.LocalMusicPlayer,
        "pause",
        lambda self: execution.ExecutionResult(
            "PAUSE", execution.ExecutionStatus.COMPLETED, "test action"
        ),
    )
    monkeypatch.setattr(
        execution.LocalMusicPlayer,
        "stop",
        lambda self: execution.ExecutionResult(
            "STOP", execution.ExecutionStatus.COMPLETED, "test action"
        ),
    )
    monkeypatch.setattr(
        execution.LocalMusicPlayer,
        "next_track",
        lambda self: execution.ExecutionResult(
            "NEXT", execution.ExecutionStatus.COMPLETED, "test action"
        ),
    )
    monkeypatch.setattr(
        execution.OutputVolume,
        "adjust",
        lambda self, intent, direction: execution.ExecutionResult(
            intent, execution.ExecutionStatus.COMPLETED, "test action"
        ),
    )
    monkeypatch.setattr(
        execution.IntentDispatcher,
        "_set_light",
        lambda self, enabled: execution.ExecutionResult(
            "LIGHT_ON" if enabled else "LIGHT_OFF",
            execution.ExecutionStatus.COMPLETED,
            "test light",
        ),
    )
    timer = FakeCountdownTimer()
    alarm = FakeAlarmScheduler()
    interaction_color_store = FakeInteractionColorStore()
    dispatcher = execution.IntentDispatcher(
        reminder_store=execution.JsonReminderStore(tmp_path / "reminders.json"),
        countdown_timer=timer,
        alarm_scheduler=alarm,
        interaction_color_store=interaction_color_store,
    )

    assert len(execution.INTENT_NAMES) == 20
    assert set(dispatcher._handlers) == set(execution.INTENT_NAMES)
    for intent_id, intent_name in enumerate(execution.INTENT_NAMES):
        result = dispatcher.execute(intent_id)
        assert result.intent == intent_name
        if intent_name == "OUT_OF_SCOPE":
            assert result.status is execution.ExecutionStatus.REJECTED
            assert (
                result.message
                == "Sorry, I didn't understand that. Please try again."
            )
        elif intent_name == "TEMPERATURE":
            assert result.status is execution.ExecutionStatus.REJECTED
            assert result.message == "Please choose 18, 22, or 26 degrees."
        elif intent_name in {
            "WEATHER",
            "CALL",
            "MESSAGE",
        }:
            assert result.status is execution.ExecutionStatus.UNAVAILABLE
        elif intent_name == "BRIGHTNESS":
            assert result.status is execution.ExecutionStatus.REJECTED
            assert result.message == "Please choose 20, 60, or 100 percent."
        elif intent_name == "COLOR":
            assert result.status is execution.ExecutionStatus.REJECTED
            assert result.message == "Please choose red, blue, or green."
        else:
            assert result.status is execution.ExecutionStatus.COMPLETED
    dispatcher.close()
    assert timer.durations == [10]
    assert timer.closed
    assert alarm.alarms == [(6, 0)]
    assert alarm.closed


@pytest.mark.parametrize(
    ("action", "seconds"),
    [
        ("10 seconds", 10),
        ("30 seconds", 30),
        ("1 minute", 60),
    ],
)
def test_timer_starts_with_selected_duration(
    action: str,
    seconds: int,
) -> None:
    timer = FakeCountdownTimer()
    dispatcher = execution.IntentDispatcher(countdown_timer=timer)

    result = dispatcher.execute("TIMER", slot_action=action)

    assert result.status is execution.ExecutionStatus.COMPLETED
    assert result.message == f"Timer set for {action}."
    assert timer.durations == [seconds]
    dispatcher.close()


def test_timer_rejects_unsupported_duration() -> None:
    dispatcher = execution.IntentDispatcher(countdown_timer=FakeCountdownTimer())

    with pytest.raises(ValueError, match="unsupported timer duration"):
        dispatcher.execute("TIMER", slot_action="2 hours")

    dispatcher.close()


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("6:00 AM", (6, 0)),
        ("8:00 AM", (8, 0)),
        ("9:00 PM", (21, 0)),
    ],
)
def test_alarm_starts_at_selected_time(action: str, expected: tuple[int, int]) -> None:
    alarm = FakeAlarmScheduler()
    dispatcher = execution.IntentDispatcher(alarm_scheduler=alarm)

    result = dispatcher.execute("ALARM", slot_action=action)

    assert result.status is execution.ExecutionStatus.COMPLETED
    assert result.message == f"Alarm set for {action}."
    assert alarm.alarms == [expected]
    dispatcher.close()


def test_alarm_rejects_unsupported_time() -> None:
    dispatcher = execution.IntentDispatcher(
        alarm_scheduler=FakeAlarmScheduler()
    )

    with pytest.raises(ValueError, match="unsupported alarm time"):
        dispatcher.execute("ALARM", slot_action="7:30 AM")

    dispatcher.close()


def test_alarm_expiry_speech_includes_local_target_time() -> None:
    speech: list[str] = []
    notify = execution.IntentDispatcher._speak_alarm_expiry(speech.append)

    notify(datetime(2026, 10, 20, 6, 0))

    assert speech == ["It's 6:00 AM."]


@pytest.mark.parametrize(
    ("action", "setpoint"),
    [
        ("18 degrees", 18),
        ("22 degrees", 22),
        ("26 degrees", 26),
    ],
)
def test_temperature_speaks_requested_setpoint_without_hardware(
    action: str,
    setpoint: int,
) -> None:
    dispatcher = execution.IntentDispatcher()

    result = dispatcher.execute("TEMPERATURE", slot_action=action)

    assert result.status is execution.ExecutionStatus.COMPLETED
    assert result.message == f"Setting the temperature to {setpoint} degrees."
    dispatcher.close()


def test_temperature_without_slot_prompts_for_a_supported_setpoint() -> None:
    dispatcher = execution.IntentDispatcher()

    result = dispatcher.execute("TEMPERATURE")

    assert result.status is execution.ExecutionStatus.REJECTED
    assert result.message == "Please choose 18, 22, or 26 degrees."
    dispatcher.close()


def test_temperature_rejects_unsupported_setpoint() -> None:
    dispatcher = execution.IntentDispatcher()

    with pytest.raises(ValueError, match="unsupported temperature set point"):
        dispatcher.execute("TEMPERATURE", slot_action="24 degrees")

    dispatcher.close()


@pytest.mark.parametrize(
    ("action", "brightness"),
    [
        ("20 percent", 20),
        ("60 percent", 60),
        ("100 percent", 100),
    ],
)
def test_brightness_updates_duocast_setting(
    action: str,
    brightness: int,
) -> None:
    updates: list[int] = []
    dispatcher = execution.IntentDispatcher(brightness_callback=updates.append)

    result = dispatcher.execute("BRIGHTNESS", slot_action=action)

    assert result.status is execution.ExecutionStatus.COMPLETED
    assert result.message == f"DuoCast brightness set to {brightness} percent."
    assert updates == [brightness]
    dispatcher.close()


def test_brightness_without_slot_prompts_for_supported_level() -> None:
    dispatcher = execution.IntentDispatcher()

    result = dispatcher.execute("BRIGHTNESS")

    assert result.status is execution.ExecutionStatus.REJECTED
    assert result.message == "Please choose 20, 60, or 100 percent."
    dispatcher.close()


def test_brightness_is_unavailable_without_duocast_rgb() -> None:
    dispatcher = execution.IntentDispatcher()

    result = dispatcher.execute("BRIGHTNESS", slot_action="60 percent")

    assert result.status is execution.ExecutionStatus.UNAVAILABLE
    assert result.message == "DuoCast RGB brightness control is not configured"
    dispatcher.close()


def test_brightness_rejects_unsupported_level() -> None:
    dispatcher = execution.IntentDispatcher(brightness_callback=lambda _: None)

    with pytest.raises(ValueError, match="unsupported DuoCast brightness level"):
        dispatcher.execute("BRIGHTNESS", slot_action="80 percent")

    dispatcher.close()


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("Red", "Red"),
        ("BLUE", "Blue"),
        ("green", "Green"),
    ],
)
def test_color_selection_is_persisted_and_updates_indicator(
    action: str,
    expected: str,
) -> None:
    store = FakeInteractionColorStore()
    updates: list[str] = []
    dispatcher = execution.IntentDispatcher(
        interaction_color_store=store,
        interaction_color_callback=updates.append,
    )

    result = dispatcher.execute("COLOR", slot_action=action)

    assert result.status is execution.ExecutionStatus.COMPLETED
    assert result.message == f"Interaction color set to {expected.lower()}."
    assert store.color == expected
    assert updates == [expected]
    dispatcher.close()


def test_color_selection_without_slot_is_rejected() -> None:
    dispatcher = execution.IntentDispatcher()

    result = dispatcher.execute("COLOR")

    assert result.status is execution.ExecutionStatus.REJECTED
    assert result.message == "Please choose red, blue, or green."
    dispatcher.close()


def test_color_selection_rejects_unsupported_color() -> None:
    dispatcher = execution.IntentDispatcher()

    with pytest.raises(ValueError, match="unsupported interaction color"):
        dispatcher.execute("COLOR", slot_action="purple")

    dispatcher.close()


def test_create_and_list_reminders_persist_across_dispatchers(tmp_path) -> None:
    store_path = tmp_path / "config" / "reminders.json"
    dispatcher = execution.IntentDispatcher(
        reminder_store=execution.JsonReminderStore(store_path)
    )

    empty_result = dispatcher.execute("LIST_REMINDERS")
    created_result = dispatcher.execute(
        "CREATE_REMINDER",
        slot_action="Drink water",
    )
    dispatcher.close()

    reloaded_dispatcher = execution.IntentDispatcher(
        reminder_store=execution.JsonReminderStore(store_path)
    )
    listed_result = reloaded_dispatcher.execute("LIST_REMINDERS")

    assert empty_result.status is execution.ExecutionStatus.COMPLETED
    assert empty_result.message == "You have no reminders."
    assert created_result.status is execution.ExecutionStatus.COMPLETED
    assert created_result.message == "Reminder added: Drink water."
    assert listed_result.status is execution.ExecutionStatus.COMPLETED
    assert listed_result.message == "You have 1 reminder: Drink water."
    reminder = execution.JsonReminderStore(store_path).list()[0]
    assert reminder.id == 1
    assert reminder.action == "Drink water"
    assert datetime.fromisoformat(reminder.created_at).utcoffset() is not None
    reloaded_dispatcher.close()


def test_create_reminder_uses_generic_action_without_a_predicted_slot(
    tmp_path,
) -> None:
    dispatcher = execution.IntentDispatcher(
        reminder_store=execution.JsonReminderStore(tmp_path / "reminders.json")
    )

    result = dispatcher.execute("CREATE_REMINDER")

    assert result.status is execution.ExecutionStatus.COMPLETED
    assert result.message == "Reminder added: Reminder."
    assert execution.JsonReminderStore(tmp_path / "reminders.json").list()[
        0
    ].action == "Reminder"
    dispatcher.close()


def test_create_reminder_rejects_unsupported_slot_action(tmp_path) -> None:
    dispatcher = execution.IntentDispatcher(
        reminder_store=execution.JsonReminderStore(tmp_path / "reminders.json")
    )

    with pytest.raises(ValueError, match="unsupported reminder action"):
        dispatcher.execute("CREATE_REMINDER", slot_action="Call mom")
    dispatcher.close()


def test_list_reminders_speaks_only_the_latest_five(tmp_path) -> None:
    store = execution.JsonReminderStore(tmp_path / "reminders.json")
    dispatcher = execution.IntentDispatcher(reminder_store=store)
    for action in execution.CREATE_REMINDER_ACTIONS:
        dispatcher.execute("CREATE_REMINDER", slot_action=action)
    for action in ("Reminder 4", "Reminder 5", "Reminder 6"):
        store.add(action)

    result = dispatcher.execute("LIST_REMINDERS")

    assert result.status is execution.ExecutionStatus.COMPLETED
    assert result.message == (
        "You have 6 reminders: Reminder 6; Reminder 5; Reminder 4; "
        "Exercise; Study; and 1 more."
    )
    dispatcher.close()


def test_local_music_player_plays_sorted_tracks_and_next_wraps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    (tmp_path / "b.mp3").touch()
    (tmp_path / "A.wav").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "c.flac").touch()
    commands: list[list[str]] = []

    class Process:
        def __init__(self) -> None:
            self.running = True

        def poll(self) -> int | None:
            return None if self.running else 0

        def terminate(self) -> None:
            self.running = False

        def wait(self, timeout: float) -> int:
            return 0

    monkeypatch.setattr(execution.shutil, "which", lambda _: "/usr/bin/ffplay")
    monkeypatch.setattr(execution.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        execution.subprocess,
        "Popen",
        lambda command, **kwargs: commands.append(command) or Process(),
    )
    player = execution.LocalMusicPlayer(tmp_path)

    assert player.play().status is execution.ExecutionStatus.COMPLETED
    assert player.next_track().status is execution.ExecutionStatus.COMPLETED
    assert player.next_track().status is execution.ExecutionStatus.COMPLETED
    assert player.next_track().status is execution.ExecutionStatus.COMPLETED

    assert [command[-1] for command in commands] == [
        str(tmp_path / "A.wav"),
        str(tmp_path / "b.mp3"),
        str(tmp_path / "subdir" / "c.flac"),
        str(tmp_path / "A.wav"),
    ]
    player.close()


def test_local_music_player_picks_up_files_added_during_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    (tmp_path / "A.wav").touch()
    (tmp_path / "b.mp3").touch()
    commands: list[list[str]] = []

    class Process:
        def __init__(self) -> None:
            self.running = True

        def poll(self) -> int | None:
            return None if self.running else 0

        def terminate(self) -> None:
            self.running = False

        def wait(self, timeout: float) -> int:
            return 0

    monkeypatch.setattr(execution.shutil, "which", lambda _: "/usr/bin/ffplay")
    monkeypatch.setattr(execution.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        execution.subprocess,
        "Popen",
        lambda command, **kwargs: commands.append(command) or Process(),
    )
    player = execution.LocalMusicPlayer(tmp_path)

    assert player.play().status is execution.ExecutionStatus.COMPLETED
    (tmp_path / "a1.mp3").touch()
    assert player.next_track().status is execution.ExecutionStatus.COMPLETED

    assert [command[-1] for command in commands] == [
        str(tmp_path / "A.wav"),
        str(tmp_path / "a1.mp3"),
    ]
    player.close()


def test_local_music_player_pause_resume_stop_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    (tmp_path / "song.mp3").touch()
    processes = []

    class Process:
        def __init__(self) -> None:
            self.return_code = None
            self.signals: list[int] = []
            self.terminated = False

        def poll(self) -> int | None:
            return self.return_code

        def send_signal(self, value: int) -> None:
            self.signals.append(value)

        def terminate(self) -> None:
            self.terminated = True
            self.return_code = -signal.SIGTERM

        def wait(self, timeout: float) -> int:
            return self.return_code or 0

        def kill(self) -> None:
            self.return_code = -signal.SIGKILL

    def start_process(*args, **kwargs) -> Process:
        process = Process()
        processes.append(process)
        return process

    monkeypatch.setattr(execution.shutil, "which", lambda _: "/usr/bin/ffplay")
    monkeypatch.setattr(execution.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        execution.subprocess, "Popen", start_process
    )
    player = execution.LocalMusicPlayer(tmp_path)

    assert player.play().status is execution.ExecutionStatus.COMPLETED
    assert player.pause().status is execution.ExecutionStatus.COMPLETED
    assert player.play().message == "music playback resumed"
    assert player.stop().status is execution.ExecutionStatus.COMPLETED

    assert processes[0].signals == [signal.SIGSTOP, signal.SIGCONT]
    assert processes[0].terminated
    player.close()


def test_local_music_player_reports_missing_tracks_or_ffplay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    player = execution.LocalMusicPlayer(tmp_path / "empty")

    result = player.play()

    assert result.status is execution.ExecutionStatus.UNAVAILABLE
    assert "no supported music files" in result.message

    (tmp_path / "song.mp3").touch()
    monkeypatch.setattr(execution.shutil, "which", lambda _: None)
    result = execution.LocalMusicPlayer(tmp_path).play()
    assert result.status is execution.ExecutionStatus.UNAVAILABLE
    assert "ffplay is not installed" in result.message


@pytest.mark.parametrize(
    ("intent", "direction", "step"),
    [
        ("VOLUME_UP", "up", "5%+"),
        ("VOLUME_DOWN", "down", "5%-"),
    ],
)
def test_volume_intents_adjust_default_output(
    monkeypatch: pytest.MonkeyPatch,
    intent: str,
    direction: str,
    step: str,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(execution.shutil, "which", lambda _: "/usr/bin/wpctl")
    monkeypatch.setattr(
        execution.subprocess,
        "run",
        lambda command, **kwargs: (
            commands.append(command) or SimpleNamespace(returncode=0, stderr=b"")
        ),
    )

    result = execution.OutputVolume().adjust(intent, direction)

    assert commands == [
        [
            "/usr/bin/wpctl",
            "set-volume",
            "--limit",
            "1.0",
            "@DEFAULT_AUDIO_SINK@",
            step,
        ]
    ]
    assert result.intent == intent
    assert result.status is execution.ExecutionStatus.COMPLETED


def test_volume_action_reports_missing_wpctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution.shutil, "which", lambda _: None)

    result = execution.OutputVolume().adjust("VOLUME_UP", "up")

    assert result.status is execution.ExecutionStatus.UNAVAILABLE
    assert "wpctl is not installed" in result.message


def test_gpio_action_requires_explicit_pin_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PI5_VCM_LIGHT_GPIO", raising=False)

    result = execution.IntentDispatcher().execute("LIGHT_ON")

    assert result.status is execution.ExecutionStatus.UNAVAILABLE
    assert "PI5_VCM_LIGHT_GPIO" in result.message


def test_gpio_action_uses_configured_output_and_releases_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, int | bool | None]] = []

    class FakeGpioLight:
        def __init__(self, pin: int) -> None:
            events.append(("init", pin))

        def set(self, enabled: bool) -> None:
            events.append(("set", enabled))

        def close(self) -> None:
            events.append(("close", None))

    monkeypatch.setenv("PI5_VCM_LIGHT_GPIO", "17")
    monkeypatch.setattr(execution, "GpioLight", FakeGpioLight)
    dispatcher = execution.IntentDispatcher()

    assert dispatcher.execute("LIGHT_ON").status is execution.ExecutionStatus.COMPLETED
    assert dispatcher.execute("LIGHT_OFF").status is execution.ExecutionStatus.COMPLETED
    dispatcher.close()

    assert events == [
        ("init", 17),
        ("set", True),
        ("set", False),
        ("close", None),
    ]


def test_gpio_action_rejects_invalid_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PI5_VCM_LIGHT_GPIO", "not-a-pin")

    with pytest.raises(ValueError, match="BCM pin"):
        execution.IntentDispatcher().execute("LIGHT_ON")


def test_time_intent_returns_local_time() -> None:
    result = execution.IntentDispatcher().execute(np.int64(2))

    assert result.intent == "TIME"
    assert result.status is execution.ExecutionStatus.COMPLETED
    assert result.message.startswith("The local time is ")


def test_weather_intent_uses_current_weather_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENWEATHER_API_KEY", "test-key")
    requests: list[tuple[str, float]] = []
    responses = [
        b'{"name":"Quezon City","main":{"temp":31.6},'
        b'"weather":[{"description":"scattered clouds"}]}'
    ]

    class Response:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def read(self) -> bytes:
            return self.body

    def fake_urlopen(request: Request, *, timeout: float) -> Response:
        requests.append((request.full_url, timeout))
        return Response(responses.pop(0))

    monkeypatch.setattr(execution, "urlopen", fake_urlopen)

    result = execution.IntentDispatcher().execute("WEATHER")

    assert result.status is execution.ExecutionStatus.COMPLETED
    assert result.message == (
        "The current weather in Quezon City, Philippines is "
        "32 degrees with scattered clouds."
    )
    assert "Celsius" not in result.message
    assert len(requests) == 1
    weather_url, timeout = requests[0]
    assert urlparse(weather_url).path == "/data/2.5/weather"
    weather_params = parse_qs(urlparse(weather_url).query)
    assert weather_params["q"] == ["Quezon City,PH"]
    assert weather_params["units"] == ["metric"]
    assert weather_params["appid"] == ["test-key"]
    assert timeout == 8
    assert not responses


def test_weather_intent_reports_missing_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    monkeypatch.setattr(
        execution,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("must not request without API key"),
    )

    result = execution.IntentDispatcher().execute("WEATHER")

    assert result.status is execution.ExecutionStatus.UNAVAILABLE
    assert result.message == (
        "Weather is unavailable right now. Please try again later."
    )


def test_weather_http_error_does_not_expose_api_key(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "private-test-api-key"
    monkeypatch.setenv("OPENWEATHER_API_KEY", secret)

    def fail_request(request: Request, *, timeout: float) -> None:
        raise HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(execution, "urlopen", fail_request)

    result = execution.IntentDispatcher().execute("WEATHER")

    assert result.status is execution.ExecutionStatus.UNAVAILABLE
    assert secret not in result.message
    assert secret not in caplog.text
    assert "HTTP 401" in caplog.text


def test_weather_intent_requests_current_conditions_each_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENWEATHER_API_KEY", "test-key")
    requests: list[str] = []
    responses = [
        b'{"main":{"temp":30},"weather":[{"description":"clear sky"}]}',
        b'{"main":{"temp":29},"weather":[{"description":"light rain"}]}',
    ]

    class Response:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def read(self) -> bytes:
            return self.body

    def fake_urlopen(request: Request, *, timeout: float) -> Response:
        requests.append(request.full_url)
        return Response(responses.pop(0))

    monkeypatch.setattr(execution, "urlopen", fake_urlopen)
    dispatcher = execution.IntentDispatcher()

    assert dispatcher.execute("WEATHER").status is execution.ExecutionStatus.COMPLETED
    assert dispatcher.execute("WEATHER").status is execution.ExecutionStatus.COMPLETED

    assert len(requests) == 2
    assert all("/data/2.5/weather" in request for request in requests)


def test_dispatcher_rejects_invalid_intents() -> None:
    dispatcher = execution.IntentDispatcher()

    with pytest.raises(ValueError, match="out of range"):
        dispatcher.execute(20)
    with pytest.raises(ValueError, match="unknown intent"):
        dispatcher.execute("NOT_AN_INTENT")
    with pytest.raises(ValueError, match="class ID"):
        dispatcher.execute(True)
