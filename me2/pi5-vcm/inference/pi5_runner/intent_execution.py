"""Dispatch ME2 intent predictions to Raspberry Pi actions."""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import isfinite
from numbers import Integral
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

if __package__:
    from .countdown_timer import CountdownTimer, TimerScheduler
    from .reminder_store import JsonReminderStore, ReminderStoreError
    from .wall_clock_alarm import AlarmScheduler, WallClockAlarm
else:
    from countdown_timer import CountdownTimer, TimerScheduler
    from reminder_store import JsonReminderStore, ReminderStoreError
    from wall_clock_alarm import AlarmScheduler, WallClockAlarm

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

OPENWEATHER_CURRENT_URL = (
    "https://api.openweathermap.org/data/2.5/weather"
)
WEATHER_LOCATION = "Quezon City,PH"
WEATHER_LOCATION_LABEL = "Quezon City, Philippines"
WEATHER_REQUEST_TIMEOUT_SECONDS = 8
MUSIC_EXTENSIONS = frozenset(
    {".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}
)
MUSIC_VOLUME_STEP = "5%"
PLAYER_STARTUP_GRACE_SECONDS = 0.2
PLAYER_STOP_TIMEOUT_SECONDS = 3
CREATE_REMINDER_ACTIONS = ("Drink water", "Study", "Exercise")
CREATE_REMINDER_SLOT_INDEX = 5
REMINDER_LIST_LIMIT = 5
TIMER_ACTIONS = ("10 seconds", "30 seconds", "1 minute")
TIMER_DURATIONS_SECONDS = (10, 30, 60)
TIMER_SLOT_INDEX = 0
TIMER_EXPIRED_MESSAGE = "Your timer is up."
ALARM_ACTIONS = ("6:00 AM", "8:00 AM", "9:00 PM")
ALARM_TIMES = ((6, 0), (8, 0), (21, 0))
ALARM_SLOT_INDEX = 1
ALARM_EXPIRED_MESSAGE_FORMAT = "It's {time}."


class WeatherServiceError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OpenWeatherClient:
    """Fetch current conditions from OpenWeather Current Weather API."""

    def current_conditions(self) -> str:
        api_key = os.environ.get("OPENWEATHER_API_KEY", "").strip()
        if not api_key:
            raise WeatherServiceError("OPENWEATHER_API_KEY is not configured")

        response = self._request_json(
            OPENWEATHER_CURRENT_URL,
            {
                "q": WEATHER_LOCATION,
                "units": "metric",
                "appid": api_key,
            },
        )
        try:
            if not isinstance(response, dict):
                raise TypeError("response must be an object")
            current = response["main"]
            if not isinstance(current, dict):
                raise TypeError("main conditions must be an object")
            temperature = current["temp"]
            weather_data = response["weather"]
            if (
                not isinstance(weather_data, list)
                or not weather_data
                or not isinstance(weather_data[0], dict)
            ):
                raise TypeError("weather data must be a non-empty object list")
            description = weather_data[0]["description"]
        except (KeyError, TypeError) as exc:
            raise WeatherServiceError(
                "OpenWeather returned an incomplete current-conditions response"
            ) from exc

        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not isfinite(temperature)
            or not isinstance(description, str)
            or not description.strip()
        ):
            raise WeatherServiceError(
                "OpenWeather returned invalid current-conditions data"
            )

        return (
            f"The current weather in {WEATHER_LOCATION_LABEL} is "
            f"{round(temperature)} degrees with {description.strip()}."
        )

    @staticmethod
    def _request_json(url: str, parameters: dict[str, object]) -> object:
        request = Request(
            f"{url}?{urlencode(parameters)}",
            headers={"User-Agent": "pi5-vcm/1.0"},
        )
        try:
            with urlopen(request, timeout=WEATHER_REQUEST_TIMEOUT_SECONDS) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            raise WeatherServiceError(
                f"OpenWeather request failed with HTTP {exc.code}",
                status_code=exc.code,
            ) from None
        except (URLError, TimeoutError, OSError) as exc:
            raise WeatherServiceError("OpenWeather request failed") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise WeatherServiceError(
                "OpenWeather returned an invalid JSON response"
            ) from exc


class LocalMusicPlayer:
    """Play a sorted local music collection using a supervised ffplay process."""

    def __init__(self, music_dir: Path | None = None) -> None:
        self.music_dir = music_dir or Path.home() / "Music"
        self._track_index = -1
        self._current_track: Path | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._paused = False

    def play(self) -> ExecutionResult:
        if self._is_running():
            if self._paused and self._process is not None:
                self._process.send_signal(signal.SIGCONT)
                self._paused = False
                return self._completed("PLAY_MUSIC", "music playback resumed")
            return self._completed("PLAY_MUSIC", "music is already playing")

        tracks = self._discover_tracks()
        if not tracks:
            return self._unavailable(
                "PLAY_MUSIC", f"no supported music files found in {self.music_dir}"
            )
        if self._current_track in tracks:
            self._track_index = tracks.index(self._current_track)
        elif self._track_index < 0 or self._track_index >= len(tracks):
            self._track_index = 0
        return self._start_track("PLAY_MUSIC", tracks)

    def pause(self) -> ExecutionResult:
        if not self._is_running() or self._process is None:
            return self._unavailable("PAUSE", "no music is currently playing")
        if not self._paused:
            self._process.send_signal(signal.SIGSTOP)
            self._paused = True
        return self._completed("PAUSE", "music playback paused")

    def stop(self) -> ExecutionResult:
        self._stop_process()
        return self._completed("STOP", "music playback stopped")

    def next_track(self) -> ExecutionResult:
        tracks = self._discover_tracks()
        if not tracks:
            return self._unavailable(
                "NEXT", f"no supported music files found in {self.music_dir}"
            )
        if self._current_track in tracks:
            self._track_index = tracks.index(self._current_track)
        elif self._current_track is not None or self._track_index >= len(tracks):
            self._track_index = -1
        self._track_index = (self._track_index + 1) % len(tracks)
        self._stop_process()
        return self._start_track("NEXT", tracks)

    def close(self) -> None:
        self._stop_process()

    def _discover_tracks(self) -> list[Path]:
        if not self.music_dir.is_dir():
            return []
        return sorted(
            (
                path
                for path in self.music_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in MUSIC_EXTENSIONS
            ),
            key=lambda path: str(path).casefold(),
        )

    def _is_running(self) -> bool:
        if self._process is None:
            return False
        if self._process.poll() is not None:
            self._process = None
            self._paused = False
            return False
        return True

    def _start_track(self, intent: str, tracks: list[Path]) -> ExecutionResult:
        executable = shutil.which("ffplay")
        if executable is None:
            return self._unavailable(
                intent,
                "ffplay is not installed; install FFmpeg for local music playback",
            )
        track = tracks[self._track_index]
        try:
            process = subprocess.Popen(
                [
                    executable,
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "error",
                    str(track),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise RuntimeError(f"could not start ffplay: {exc}") from exc
        time.sleep(PLAYER_STARTUP_GRACE_SECONDS)
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"ffplay exited during startup with status {return_code}"
            )
        self._process = process
        self._paused = False
        self._current_track = track
        return self._completed(intent, f"playing {track.name}")

    def _stop_process(self) -> None:
        process = self._process
        was_paused = self._paused
        self._process = None
        self._paused = False
        if process is None or process.poll() is not None:
            return
        if was_paused:
            process.send_signal(signal.SIGCONT)
        process.terminate()
        try:
            process.wait(timeout=PLAYER_STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=PLAYER_STOP_TIMEOUT_SECONDS)

    @staticmethod
    def _completed(intent: str, message: str) -> ExecutionResult:
        return ExecutionResult(intent, ExecutionStatus.COMPLETED, message)

    @staticmethod
    def _unavailable(intent: str, message: str) -> ExecutionResult:
        return ExecutionResult(intent, ExecutionStatus.UNAVAILABLE, message)


class OutputVolume:
    """Adjust the default PipeWire output, including music and spoken replies."""

    def adjust(self, intent: str, direction: str) -> ExecutionResult:
        executable = shutil.which("wpctl")
        if executable is None:
            return ExecutionResult(
                intent,
                ExecutionStatus.UNAVAILABLE,
                "wpctl is not installed; cannot adjust output volume",
            )
        adjustment = f"{MUSIC_VOLUME_STEP}{'+' if direction == 'up' else '-'}"
        try:
            completed = subprocess.run(
                [
                    executable,
                    "set-volume",
                    "--limit",
                    "1.0",
                    "@DEFAULT_AUDIO_SINK@",
                    adjustment,
                ],
                capture_output=True,
                check=False,
                timeout=5,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("wpctl timed out while adjusting output volume") from exc
        if completed.returncode != 0:
            details = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"wpctl set-volume failed with status {completed.returncode}"
                + (f": {details}" if details else "")
            )
        return ExecutionResult(
            intent,
            ExecutionStatus.COMPLETED,
            f"default output volume adjusted {direction} by 5%",
        )


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

    def __init__(
        self,
        reminder_store: JsonReminderStore | None = None,
        *,
        timer_expiry_callback: Callable[[], None] | None = None,
        countdown_timer: TimerScheduler | None = None,
        alarm_expiry_callback: Callable[[str], None] | None = None,
        alarm_scheduler: AlarmScheduler | None = None,
    ) -> None:
        self._gpio_light: GpioLight | None = None
        self._weather_client = OpenWeatherClient()
        self._music_player = LocalMusicPlayer()
        self._output_volume = OutputVolume()
        self._reminder_store = (
            reminder_store if reminder_store is not None else JsonReminderStore()
        )
        self._countdown_timer = (
            countdown_timer
            if countdown_timer is not None
            else CountdownTimer(timer_expiry_callback or self._log_timer_expiry)
        )
        self._alarm_scheduler = (
            alarm_scheduler
            if alarm_scheduler is not None
            else WallClockAlarm(
                self._speak_alarm_expiry(alarm_expiry_callback)
            )
        )
        self._handlers = {
            "PLAY_MUSIC": self._music_player.play,
            "WEATHER": self._tell_weather,
            "TIME": self._tell_time,
            "LIGHT_ON": lambda: self._set_light(True),
            "LIGHT_OFF": lambda: self._set_light(False),
            "PAUSE": self._music_player.pause,
            "STOP": self._music_player.stop,
            "NEXT": self._music_player.next_track,
            "VOLUME_UP": lambda: self._output_volume.adjust("VOLUME_UP", "up"),
            "VOLUME_DOWN": lambda: self._output_volume.adjust("VOLUME_DOWN", "down"),
            "CALL": lambda: self._unavailable(
                "CALL", "a telephony provider is not configured"
            ),
            "MESSAGE": lambda: self._unavailable(
                "MESSAGE", "a messaging provider is not configured"
            ),
            "LIST_REMINDERS": self._list_reminders,
            "TIMER": lambda: self._set_timer(None),
            "ALARM": lambda: self._set_alarm(None),
            "TEMPERATURE": lambda: self._unavailable(
                "TEMPERATURE", "a temperature sensor is not configured"
            ),
            "BRIGHTNESS": lambda: self._unavailable(
                "BRIGHTNESS", "a display brightness controller is not configured"
            ),
            "COLOR": lambda: self._unavailable(
                "COLOR", "a color-capable lighting controller is not configured"
            ),
            "CREATE_REMINDER": lambda: self._create_reminder(None),
            "OUT_OF_SCOPE": lambda: self._rejected(
                "OUT_OF_SCOPE",
                "Sorry, I didn't understand that. Please try again.",
            ),
        }

    def execute(
        self,
        intent: str | int,
        *,
        slot_action: str | None = None,
    ) -> ExecutionResult:
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
        if intent_name == "CREATE_REMINDER":
            result = self._create_reminder(slot_action)
        elif intent_name == "TIMER":
            result = self._set_timer(slot_action)
        elif intent_name == "ALARM":
            result = self._set_alarm(slot_action)
        else:
            result = handler()
        logger.info(
            "intent execution: intent=%s status=%s message=%s",
            result.intent,
            result.status,
            result.message,
        )
        return result

    def close(self) -> None:
        self._countdown_timer.close()
        self._alarm_scheduler.close()
        self._music_player.close()
        if self._gpio_light is not None:
            self._gpio_light.close()
            self._gpio_light = None

    def _tell_weather(self) -> ExecutionResult:
        try:
            message = self._weather_client.current_conditions()
        except WeatherServiceError as exc:
            if exc.status_code is not None:
                logger.warning(
                    "weather request failed: HTTP %d",
                    exc.status_code,
                )
            else:
                logger.warning("weather request unavailable: %s", exc)
            return self._unavailable(
                "WEATHER",
                "Weather is unavailable right now. Please try again later.",
            )
        return ExecutionResult(
            intent="WEATHER",
            status=ExecutionStatus.COMPLETED,
            message=message,
        )

    def _create_reminder(self, slot_action: str | None) -> ExecutionResult:
        action = "Reminder" if slot_action is None else slot_action
        if slot_action is not None and slot_action not in CREATE_REMINDER_ACTIONS:
            raise ValueError(f"unsupported reminder action: {slot_action!r}")
        try:
            reminder = self._reminder_store.add(action)
        except ReminderStoreError:
            logger.exception("could not persist a reminder")
            raise
        return ExecutionResult(
            intent="CREATE_REMINDER",
            status=ExecutionStatus.COMPLETED,
            message=f"Reminder added: {reminder.action}.",
        )

    def _set_timer(self, slot_action: str | None) -> ExecutionResult:
        action = TIMER_ACTIONS[0] if slot_action is None else slot_action
        if action not in TIMER_ACTIONS:
            raise ValueError(f"unsupported timer duration: {action!r}")
        duration_seconds = TIMER_DURATIONS_SECONDS[TIMER_ACTIONS.index(action)]
        self._countdown_timer.start(duration_seconds)
        return ExecutionResult(
            intent="TIMER",
            status=ExecutionStatus.COMPLETED,
            message=f"Timer set for {action}.",
        )

    def _set_alarm(self, slot_action: str | None) -> ExecutionResult:
        action = ALARM_ACTIONS[0] if slot_action is None else slot_action
        if action not in ALARM_ACTIONS:
            raise ValueError(f"unsupported alarm time: {action!r}")
        hour, minute = ALARM_TIMES[ALARM_ACTIONS.index(action)]
        self._alarm_scheduler.start_alarm(hour, minute)
        return ExecutionResult(
            intent="ALARM",
            status=ExecutionStatus.COMPLETED,
            message=f"Alarm set for {action}.",
        )

    @staticmethod
    def _speak_alarm_expiry(
        callback: Callable[[str], None] | None,
    ) -> Callable[[datetime], None]:
        def notify(target: datetime) -> None:
            if callback is None:
                logger.warning("alarm expired but no speech callback is configured")
                return
            alarm_time = target.strftime("%I:%M %p").lstrip("0")
            callback(ALARM_EXPIRED_MESSAGE_FORMAT.format(time=alarm_time))

        return notify

    @staticmethod
    def _log_timer_expiry() -> None:
        logger.warning("timer expired but no speech callback is configured")

    def _list_reminders(self) -> ExecutionResult:
        try:
            reminders = self._reminder_store.list()
        except ReminderStoreError:
            logger.exception("could not list reminders")
            raise
        if not reminders:
            return ExecutionResult(
                intent="LIST_REMINDERS",
                status=ExecutionStatus.COMPLETED,
                message="You have no reminders.",
            )

        visible_actions = [
            reminder.action for reminder in reminders[:REMINDER_LIST_LIMIT]
        ]
        remainder = len(reminders) - len(visible_actions)
        action_list = "; ".join(visible_actions)
        if remainder:
            action_list += f"; and {remainder} more"
        item_word = "reminder" if len(reminders) == 1 else "reminders"
        return ExecutionResult(
            intent="LIST_REMINDERS",
            status=ExecutionStatus.COMPLETED,
            message=f"You have {len(reminders)} {item_word}: {action_list}.",
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
