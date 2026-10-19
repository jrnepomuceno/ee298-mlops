"""Dispatch ME2 intent predictions to Raspberry Pi actions."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from numbers import Integral

INTENT_NAMES = (
    "PLAY_MUSIC",
    "WEATHER",
    "TIME",
    "LIGHT_ON",
    "LIGHT_OFF",
    "PAUSE",
    "STOP",
    "NEXT",
    "VOLUME_UP",
    "VOLUME_DOWN",
    "CALL",
    "MESSAGE",
    "LIST_REMINDERS",
    "TIMER",
    "ALARM",
    "TEMPERATURE",
    "BRIGHTNESS",
    "COLOR",
    "CREATE_REMINDER",
    "OUT_OF_SCOPE",
)

logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ExecutionResult:
    intent: str
    status: ExecutionStatus
    message: str


class GpioLight:
    """Control one configured GPIO output for a light or relay module."""

    def __init__(self, pin: int) -> None:
        try:
            from gpiozero import LED
        except ImportError as exc:
            raise RuntimeError(
                "gpiozero is required for GPIO light control; install it on the Pi"
            ) from exc
        self._output = LED(pin)

    def set(self, enabled: bool) -> None:
        if enabled:
            self._output.on()
        else:
            self._output.off()

    def close(self) -> None:
        self._output.close()


class IntentDispatcher:
    """Execute supported local actions for a predicted ME2 intent.

    Call ``close`` when the Pi inference service shuts down to release GPIO.
    The caller is responsible for applying the exported model's OOS threshold
    before dispatching an intent.
    """

    def __init__(self) -> None:
        self._gpio_light: GpioLight | None = None
        self._handlers = {
            "PLAY_MUSIC": lambda: self._player_action("PLAY_MUSIC", "play"),
            "WEATHER": lambda: self._unavailable(
                "WEATHER", "a weather provider is not configured"
            ),
            "TIME": self._tell_time,
            "LIGHT_ON": lambda: self._set_light(True),
            "LIGHT_OFF": lambda: self._set_light(False),
            "PAUSE": lambda: self._player_action("PAUSE", "pause"),
            "STOP": lambda: self._player_action("STOP", "stop"),
            "NEXT": lambda: self._player_action("NEXT", "next"),
            "VOLUME_UP": lambda: self._player_action(
                "VOLUME_UP", "volume", "0.05+"
            ),
            "VOLUME_DOWN": lambda: self._player_action(
                "VOLUME_DOWN", "volume", "0.05-"
            ),
            "CALL": lambda: self._unavailable(
                "CALL", "a telephony provider is not configured"
            ),
            "MESSAGE": lambda: self._unavailable(
                "MESSAGE", "a messaging provider is not configured"
            ),
            "LIST_REMINDERS": lambda: self._unavailable(
                "LIST_REMINDERS", "a reminder store is not configured"
            ),
            "TIMER": lambda: self._unavailable(
                "TIMER", "a timer service is not configured"
            ),
            "ALARM": lambda: self._unavailable(
                "ALARM", "an alarm service is not configured"
            ),
            "TEMPERATURE": lambda: self._unavailable(
                "TEMPERATURE", "a temperature sensor is not configured"
            ),
            "BRIGHTNESS": lambda: self._unavailable(
                "BRIGHTNESS", "a display brightness controller is not configured"
            ),
            "COLOR": lambda: self._unavailable(
                "COLOR", "a color-capable lighting controller is not configured"
            ),
            "CREATE_REMINDER": lambda: self._unavailable(
                "CREATE_REMINDER", "a reminder store is not configured"
            ),
            "OUT_OF_SCOPE": lambda: self._rejected(
                "OUT_OF_SCOPE", "out-of-scope predictions are not executed"
            ),
        }

    def execute(self, intent: str | int) -> ExecutionResult:
        """Dispatch a class ID or label from the model's 20-intent contract."""
        if isinstance(intent, bool):
            raise ValueError("intent must be a class ID or intent label")
        if isinstance(intent, Integral):
            intent_id = int(intent)
            if not 0 <= intent_id < len(INTENT_NAMES):
                raise ValueError(f"intent class ID is out of range: {intent_id}")
            intent_name = INTENT_NAMES[intent_id]
        elif isinstance(intent, str):
            intent_name = intent
        else:
            raise ValueError("intent must be a class ID or intent label")

        handler = self._handlers.get(intent_name)
        if handler is None:
            raise ValueError(f"unknown intent label: {intent_name!r}")
        result = handler()
        logger.info(
            "intent execution: intent=%s status=%s message=%s",
            result.intent,
            result.status,
            result.message,
        )
        return result

    def close(self) -> None:
        if self._gpio_light is not None:
            self._gpio_light.close()
            self._gpio_light = None

    def _player_action(self, intent: str, *arguments: str) -> ExecutionResult:
        executable = shutil.which("playerctl")
        if executable is None:
            return self._unavailable(
                intent,
                "playerctl is not installed; install it to control media players",
            )
        try:
            completed = subprocess.run(
                [executable, *arguments],
                capture_output=True,
                check=False,
                timeout=5,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "playerctl timed out while executing a media action"
            ) from exc
        if completed.returncode != 0:
            details = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"playerctl {arguments[0]} failed with status "
                f"{completed.returncode}"
                + (f": {details}" if details else "")
            )
        return ExecutionResult(
            intent=intent,
            status=ExecutionStatus.COMPLETED,
            message=f"media action completed: {arguments[0]}",
        )

    def _set_light(self, enabled: bool) -> ExecutionResult:
        raw_pin = os.environ.get("PI5_VCM_LIGHT_GPIO")
        if raw_pin is None:
            return self._unavailable(
                "LIGHT_ON" if enabled else "LIGHT_OFF",
                "set PI5_VCM_LIGHT_GPIO to the BCM pin connected to a light relay",
            )
        try:
            pin = int(raw_pin)
        except ValueError as exc:
            raise ValueError("PI5_VCM_LIGHT_GPIO must be a BCM pin number") from exc
        if pin < 0:
            raise ValueError("PI5_VCM_LIGHT_GPIO must be a non-negative BCM pin")
        if self._gpio_light is None:
            self._gpio_light = GpioLight(pin)
        self._gpio_light.set(enabled)
        intent = "LIGHT_ON" if enabled else "LIGHT_OFF"
        return ExecutionResult(
            intent=intent,
            status=ExecutionStatus.COMPLETED,
            message=f"GPIO light output {intent.lower().replace('_', ' ')}",
        )

    @staticmethod
    def _tell_time() -> ExecutionResult:
        time_text = datetime.now().strftime("%I:%M %p").lstrip("0")
        return ExecutionResult(
            intent="TIME",
            status=ExecutionStatus.COMPLETED,
            message=f"The local time is {time_text}.",
        )

    @staticmethod
    def _unavailable(intent: str, reason: str) -> ExecutionResult:
        return ExecutionResult(
            intent=intent,
            status=ExecutionStatus.UNAVAILABLE,
            message=reason,
        )

    @staticmethod
    def _rejected(intent: str, reason: str) -> ExecutionResult:
        return ExecutionResult(
            intent=intent,
            status=ExecutionStatus.REJECTED,
            message=reason,
        )
