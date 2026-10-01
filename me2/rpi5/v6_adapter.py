"""Map v6 training labels/slots to the existing runtime contract safely."""
from __future__ import annotations

from typing import Any


V6_TO_RUNTIME_INTENT = {
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

V6_ACTION_LABELS = frozenset({
    "PLAY_MUSIC", "PAUSE", "STOP", "TIME", "VOLUME_UP", "VOLUME_DOWN",
})


def adapt_v6_result(result: dict[str, Any], contract: dict[str, Any],
                    slot_threshold: float = 0.75) -> dict[str, Any]:
    """Translate a v6 prediction for the facade; unknown classes reject safely."""
    selected_model_intent = str(result.get("intent", ""))
    model_intent = str(result.get("pre_threshold_intent", selected_model_intent))
    runtime_intent = V6_TO_RUNTIME_INTENT.get(selected_model_intent)
    slots: dict[str, Any] = {}
    raw_slots = result.get("slots") or {}

    def slot_value(name: str) -> Any | None:
        prediction = raw_slots.get(name)
        if not isinstance(prediction, dict):
            return None
        if float(prediction.get("confidence", 0.0)) < slot_threshold:
            return None
        return prediction.get("value")

    if runtime_intent == "set_timer":
        minutes = slot_value("timer_minute")
        seconds = slot_value("timer_second")
        if isinstance(minutes, int) and isinstance(seconds, int):
            if seconds == 0:
                slots = {"duration": minutes, "duration_unit": "minute"}
            else:
                slots = {"duration": minutes * 60 + seconds,
                         "duration_unit": "second"}
    elif runtime_intent == "set_alarm":
        hour = slot_value("alarm_hour")
        minute = slot_value("alarm_minute")
        meridiem = slot_value("alarm_meridiem")
        if (isinstance(hour, int) and isinstance(minute, int)
                and meridiem in {"AM", "PM"}):
            slots = {"time": f"{hour}:{minute:02d} {meridiem}"}
    elif runtime_intent == "set_temperature":
        degrees = slot_value("degrees")
        if isinstance(degrees, int):
            slots = {"temperature": degrees}
    elif runtime_intent == "dim_lights":
        percent = slot_value("percent")
        if isinstance(percent, int):
            slots = {"percent": percent}
    elif runtime_intent == "remind":
        task = slot_value("task")
        if isinstance(task, str):
            slots = {"note": task}

    adapted = {**result}
    adapted["model_intent"] = model_intent
    adapted["model_slots"] = raw_slots
    adapted["intent"] = ("oov" if selected_model_intent == "oov"
                         else runtime_intent or "oov")
    adapted["slots"] = slots
    return adapted


def adapt_v6_action_result(result: dict[str, Any], contract: dict[str, Any],
                           slot_threshold: float = 0.75) -> dict[str, Any]:
    """Expose only reviewed media and volume labels to action execution."""
    adapted = adapt_v6_result(result, contract, slot_threshold)
    if (result.get("intent") not in V6_ACTION_LABELS
            or adapted["intent"] == "oov"):
        adapted["intent"] = "oov"
        adapted["slots"] = {}
    return adapted