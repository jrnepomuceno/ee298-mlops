from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from inference.pi5_runner import intent_execution as execution


def test_dispatcher_covers_all_model_intents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        execution.IntentDispatcher,
        "_player_action",
        lambda self, intent, *args: execution.ExecutionResult(
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
    dispatcher = execution.IntentDispatcher()

    assert len(execution.INTENT_NAMES) == 20
    assert set(dispatcher._handlers) == set(execution.INTENT_NAMES)
    for intent_id, intent_name in enumerate(execution.INTENT_NAMES):
        result = dispatcher.execute(intent_id)
        assert result.intent == intent_name
        if intent_name == "OUT_OF_SCOPE":
            assert result.status is execution.ExecutionStatus.REJECTED
        elif intent_name in {
            "WEATHER",
            "CALL",
            "MESSAGE",
            "LIST_REMINDERS",
            "TIMER",
            "ALARM",
            "TEMPERATURE",
            "BRIGHTNESS",
            "COLOR",
            "CREATE_REMINDER",
        }:
            assert result.status is execution.ExecutionStatus.UNAVAILABLE
        else:
            assert result.status is execution.ExecutionStatus.COMPLETED


@pytest.mark.parametrize(
    ("intent", "arguments"),
    [
        ("PLAY_MUSIC", ["play"]),
        ("PAUSE", ["pause"]),
        ("STOP", ["stop"]),
        ("NEXT", ["next"]),
        ("VOLUME_UP", ["volume", "0.05+"]),
        ("VOLUME_DOWN", ["volume", "0.05-"]),
    ],
)
def test_media_intents_call_playerctl(
    monkeypatch: pytest.MonkeyPatch,
    intent: str,
    arguments: list[str],
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(execution.shutil, "which", lambda name: "/usr/bin/playerctl")
    monkeypatch.setattr(
        execution.subprocess,
        "run",
        lambda command, **kwargs: (
            commands.append(command) or SimpleNamespace(returncode=0, stderr=b"")
        ),
    )

    result = execution.IntentDispatcher().execute(intent)

    assert commands == [["/usr/bin/playerctl", *arguments]]
    assert result.intent == intent
    assert result.status is execution.ExecutionStatus.COMPLETED


def test_media_intent_reports_missing_playerctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution.shutil, "which", lambda _: None)

    result = execution.IntentDispatcher().execute("PLAY_MUSIC")

    assert result.status is execution.ExecutionStatus.UNAVAILABLE
    assert "playerctl is not installed" in result.message


def test_media_command_failure_is_not_reported_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution.shutil, "which", lambda _: "/usr/bin/playerctl")
    monkeypatch.setattr(
        execution.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stderr=b"No players found"
        ),
    )

    with pytest.raises(RuntimeError, match="No players found"):
        execution.IntentDispatcher().execute("PAUSE")


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


def test_dispatcher_rejects_invalid_intents() -> None:
    dispatcher = execution.IntentDispatcher()

    with pytest.raises(ValueError, match="out of range"):
        dispatcher.execute(20)
    with pytest.raises(ValueError, match="unknown intent"):
        dispatcher.execute("NOT_AN_INTENT")
    with pytest.raises(ValueError, match="class ID"):
        dispatcher.execute(True)
