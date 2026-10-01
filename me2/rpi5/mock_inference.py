"""Deterministic model substitute for model-free runtime scenario testing."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .facade import INTENT_SPECS

MOCK_SLOTS: dict[str, dict[str, Any]] = {
    "turn_on_lights": {},
    "turn_off_lights": {},
    "dim_lights": {"percent": 40},
    "set_temperature": {"temperature": 22},
    "play_music": {},
    "next_music": {},
    "pause_music": {},
    "stop_music": {},
    "set_timer": {"duration": 5, "duration_unit": "minute"},
    "set_alarm": {"time": "7:30 AM"},
    "stop_timer": {},
    "remind": {"note": "water the plants"},
    "call": {"contact": "mom"},
    "what_time": {},
    "what_weather": {},
    "what_reminders": {},
    "volume_up": {},
    "volume_down": {},
    "oov": {},
}

if set(MOCK_SLOTS) != set(INTENT_SPECS) | {"oov"}:
    raise RuntimeError("mock scenario catalog must cover every facade intent plus oov")


class MockInference:
    """Return known-good intent/slot results without loading an ONNX model."""

    def recognize(self, intent: str, confidence: float = 0.99,
                  slot_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        if intent not in MOCK_SLOTS:
            raise ValueError(f"no mock scenario for intent {intent!r}")
        slots = deepcopy(MOCK_SLOTS[intent])
        if slot_overrides:
            slots.update(slot_overrides)
        return {
            "intent": intent,
            "intent_confidence": confidence,
            "slots": slots,
            "transcript": f"[mock:{intent}]",
            "frames": 0,
            "latency_ms": 0.0,
            "backend": "mock",
        }
