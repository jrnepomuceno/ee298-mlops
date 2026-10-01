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
    "next_music": "next_music.wav",
    "pause_music": "pause_music.wav",
    "stop_music": "stop_music.wav",
    "set_timer": "set_timer.wav",
    "set_alarm": "set_alarm.wav",
    "stop_timer": "stop_timer.wav",
    "remind": "remind.wav",
    "call": "call.wav",
    "what_time": "what_time.wav",
    "what_weather": "what_weather.wav",
    "what_reminders": "what_reminders.wav",
    "volume_up": "volume_up.wav",
    "volume_down": "volume_down.wav",
    "oov": "oov.wav",
}


def _action_view(action: Any) -> dict[str, Any]:
    """Normalise an action outcome into a flat dict for reply selection.

    ``action`` arrives in one of two shapes depending on the execution path:

    * a plain ``dict`` from the legacy :class:`~rpi5.harness.DryRunDispatcher`
      (keys ``status`` / ``code`` / ``action``), or
    * an :class:`~rpi5.orchestrator.OrchestratorResult` dataclass from the
      facade path (attributes ``handled`` / ``reject`` / ``request`` /
      ``execution``).

    Both reduce to the same ``{"status", "code"}`` view that
    :func:`reply_wav_name` and :func:`build_reply` select on. Treating a
    non-dict ``action`` as ``{}`` (the old behaviour) made an OOV utterance
    crash with ``AttributeError: 'OrchestratorResult' object has no attribute
    'get'`` because ``OrchestratorResult`` has no ``.get``.
    """
    if action is None:
        return {}
    if isinstance(action, dict):
        return action
    # OrchestratorResult (or any object exposing these attributes).
    reject = getattr(action, "reject", None)
    if reject is not None:
        # Map the facade reason onto the legacy action code that the reply
        # selectors key on, so an OOV / low-confidence rejection picks oov.wav
        # exactly as the old dict-shaped path did.
        reason = getattr(reject, "reason", "")
        code = {
            "oov": "out_of_vocabulary",
            "low_confidence": "confidence_below_threshold",
        }.get(reason, reason)
        return {"status": "rejected", "code": code}
    request = getattr(action, "request", None)
    if request is not None:
        return {"status": "dry_run", "code": getattr(request, "action_code", "")}
    return {}


def reply_wav_name(result: dict[str, Any] | None, action: Any) -> str:
    """Select a static reply WAV for a recognition event."""
    result = result or {}
    view = _action_view(action)
    if view.get("code") == "out_of_vocabulary":
        return "oov.wav"
    if view.get("code") == "confidence_below_threshold":
        return "oov.wav"
    return REPLY_WAV_BY_INTENT.get(result.get("intent", "oov"), "oov.wav")


def build_reply(result: dict[str, Any] | None, action: Any) -> dict[str, Any]:
    """Build a truthful spoken/display reply without an LLM or ASR."""
    result = result or {}
    view = _action_view(action)
    if view.get("status") == "rejected":
        if view.get("code") == "invalid_slot":
            text = "Invalid value. Try again."
        else:
            text = "I don't understand."
        return {"text": text, "speak": True, "source": "template"}

    intent = result.get("intent", "unknown")
    slots = result.get("slots") or {}
    prefix = "I would " if view.get("status") == "dry_run" else ""
    templates = {
        "turn_on_lights": f"{prefix}turn the lights on.",
        "turn_off_lights": f"{prefix}turn the lights off.",
        "dim_lights": f"{prefix}dim the lights to {slots.get('percent', 'that')} percent.",
        "set_temperature": f"Temperature set to {slots.get('temperature', 'that')} degrees.",
        "play_music": f"{prefix}play music.",
        "next_music": f"{prefix}skip to the next song.",
        "pause_music": f"{prefix}pause the music.",
        "stop_music": f"{prefix}stop the music.",
        "set_timer": _timer_reply(prefix, slots),
        "set_alarm": f"{prefix}set the alarm for {slots.get('time', 'that time')}.",
        "stop_timer": f"{prefix}stop the timer.",
        "remind": f"{prefix}remind you to {slots.get('note', 'do that')}.",
        "call": f"{prefix}call {slots.get('contact', 'that contact')}.",
        "what_time": f"The time is {slots.get('time', _local_time_text())}.",
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
