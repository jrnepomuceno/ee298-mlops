"""Deterministic, offline assistant replies for recognized commands."""
from __future__ import annotations

from datetime import datetime
from typing import Any


REPLY_WAV_BY_INTENT = {
    "turn_on_lights": "turn_on_lights.wav",
    "turn_off_lights": "turn_off_lights.wav",
    "dim_lights": "dim_lights.wav",
    "set_temperature": "set_temperature.wav",
    "play_music": "play_music.wav",
    "pause_music": "pause_music.wav",
    "stop_music": "stop_music.wav",
    "set_timer": "set_timer.wav",
    "set_alarm": "set_alarm.wav",
    "cancel_timer": "cancel_timer.wav",
    "remind": "remind.wav",
    "call": "call.wav",
    "what_time": "what_time.wav",
    "what_weather": "what_weather.wav",
    "what_reminders": "what_reminders.wav",
    "oov": "oov.wav",
}


def reply_wav_name(result: dict[str, Any] | None, action: dict[str, Any] | None) -> str:
    """Select a static reply WAV for a recognition event."""
    result = result or {}
    action = action or {}
    if action.get("code") == "out_of_vocabulary":
        return "oov.wav"
    if action.get("code") == "confidence_below_threshold":
        return "oov.wav"
    return REPLY_WAV_BY_INTENT.get(result.get("intent", "oov"), "oov.wav")


def build_reply(result: dict[str, Any] | None, action: dict[str, Any] | None) -> dict[str, Any]:
    """Build a truthful spoken/display reply without an LLM or ASR."""
    result = result or {}
    action = action or {}
    if action.get("status") == "rejected":
        if action.get("code") == "out_of_vocabulary":
            text = "I did not recognize that command."
        elif action.get("code") == "confidence_below_threshold":
            text = "I am not confident I understood that."
        else:
            text = "I cannot perform that command yet."
        return {"text": text, "speak": True, "source": "template"}

    intent = result.get("intent", "unknown")
    slots = result.get("slots") or {}
    prefix = "I would " if action.get("status") == "dry_run" else ""
    templates = {
        "turn_on_lights": f"{prefix}turn the lights on.",
        "turn_off_lights": f"{prefix}turn the lights off.",
        "dim_lights": f"{prefix}dim the lights to {slots.get('percent', 'that')} percent.",
        "set_temperature": f"{prefix}set the temperature to {slots.get('temperature', 'that')} degrees.",
        "play_music": f"{prefix}play music.",
        "pause_music": f"{prefix}pause the music.",
        "stop_music": f"{prefix}stop the music.",
        "set_timer": _timer_reply(prefix, slots),
        "set_alarm": f"{prefix}set the alarm for {slots.get('time', 'that time')}.",
        "cancel_timer": f"{prefix}cancel the timer.",
        "remind": f"{prefix}remind you to {slots.get('note', 'do that')}.",
        "call": f"{prefix}call {slots.get('contact', 'that contact')}.",
        "what_time": f"The time is {_local_time_text()}.",
        "what_weather": "Weather is not available without a configured local source.",
        "what_reminders": "Your local reminders are available.",
    }
    text = templates.get(intent, "I recognized the command, but cannot reply to it yet.")
    return {"text": text, "speak": True, "source": "template"}


def _timer_reply(prefix: str, slots: dict[str, Any]) -> str:
    duration = slots.get("duration")
    unit = slots.get("duration_unit", "minutes")
    if duration is None:
        return f"{prefix}set the timer."
    return f"{prefix}set a timer for {duration} {unit}."


def _local_time_text() -> str:
    """Return the host's local time in concise speech-friendly form."""
    text = datetime.now().astimezone().strftime("%I:%M %p")
    return text.lstrip("0")
