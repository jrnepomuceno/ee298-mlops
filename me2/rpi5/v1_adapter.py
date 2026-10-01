"""Map v1 intent-only training labels to the existing runtime contract safely."""
from __future__ import annotations

from typing import Any


V1_TO_RUNTIME_INTENT = {
    "PLAY_MUSIC": "play_music",
    "WEATHER": "what_weather",
    "TIME": "what_time",
    "LIGHT_ON": "turn_on_lights",
    "LIGHT_OFF": "turn_off_lights",
    "PAUSE": "pause_music",
    "STOP": "stop_music",
    "NEXT": "next_music",
    "VOLUME_UP": "volume_up",
    "VOLUME_DOWN": "volume_down",
    "CALL": "call",
    "LIST_REMINDERS": "what_reminders",
    "TIMER": "set_timer",
    "ALARM": "set_alarm",
    "TEMPERATURE": "set_temperature",
    "BRIGHTNESS": "dim_lights",
    "CREATE_REMINDER": "remind",
}

V1_ACTION_LABELS = frozenset({
    "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF", "PAUSE",
    "STOP", "VOLUME_UP", "VOLUME_DOWN", "LIST_REMINDERS", "TIMER",
    "ALARM", "TEMPERATURE", "BRIGHTNESS", "CREATE_REMINDER",
})

DEFAULT_V1_SLOTS: dict[str, dict[str, Any]] = {
    "set_timer": {"duration": 5, "duration_unit": "minute"},
    "set_alarm": {"time": "7:00 AM"},
    "set_temperature": {"temperature": 22},
    "dim_lights": {"percent": 50},
    "remind": {"note": "check reminders"},
}


def adapt_v1_result(result: dict[str, Any], contract: dict[str, Any] | None = None,
                    default_slots: bool = True) -> dict[str, Any]:
    """Translate a v1 intent-only prediction for the facade."""
    selected_model_intent = str(result.get("intent", ""))
    model_intent = str(result.get("pre_threshold_intent", selected_model_intent))
    runtime_intent = V1_TO_RUNTIME_INTENT.get(selected_model_intent)
    slots: dict[str, Any] = {}
    if default_slots and runtime_intent in DEFAULT_V1_SLOTS:
        slots = dict(DEFAULT_V1_SLOTS[runtime_intent])

    adapted = {**result}
    adapted["model_intent"] = model_intent
    adapted["model_slots"] = {}
    adapted["intent"] = ("oov" if selected_model_intent == "oov"
                         else runtime_intent or "oov")
    adapted["slots"] = slots
    return adapted


def adapt_v1_action_result(result: dict[str, Any], contract: dict[str, Any] | None = None,
                           default_slots: bool = True) -> dict[str, Any]:
    """Expose v1 labels backed by active, validated project intents."""
    adapted = adapt_v1_result(result, contract, default_slots=default_slots)
    if (result.get("intent") not in V1_ACTION_LABELS
            or adapted["intent"] == "oov"):
        adapted["intent"] = "oov"
        adapted["slots"] = {}
    return adapted
