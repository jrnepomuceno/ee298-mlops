"""Thermostat / temperature set.

Audio-only for now: the recognized slot is validated against the device's
sane range and confirmed back by voice. There is no relay or controller
wired yet, so nothing physical is touched -- the "side effect" is the spoken
confirmation plus a structured ``payload`` a future controller can consume.

The slot arrives as ``temperature`` (see :mod:`model.slots` and the facade's
``INTENT_SPECS`` for ``set_temperature``). The valid band is 10-35 degrees;
the facade already rejects out-of-band values as ``invalid_slot``, so the
executor re-checks only as a defensive guard and never trusts the slot
blindly.
"""
from __future__ import annotations

from ..facade import ActionRequest
from .base import ExecutionResult, Executor

#: Inclusive set-point band the device accepts, in degrees.
TEMP_MIN = 10
TEMP_MAX = 35


def _spoken_set(temp: int) -> str:
    """TTS-friendly confirmation for a set-point change."""
    return f"Setting the temperature to {int(temp)} degrees."


class HvacExecutor(Executor):
    """Confirms a temperature set-point by voice.

    Dry-run behaves like the base (no side effects). With ``dry_run=False``
    it validates the slot, records the requested set-point in the payload for
    a future controller to act on, and returns the spoken confirmation.
    """

    category = "hvac"

    def _live(self, req: ActionRequest) -> ExecutionResult:
        if req.action_code != "thermostat.set":
            return ExecutionResult(ok=False, intent=req.intent, category=self.category,
                                   action_code=req.action_code,
                                   detail=f"unsupported hvac action {req.action_code}",
                                   side_effects=False)
        raw = req.slots.get("temperature")
        try:
            temp = int(raw)
        except (TypeError, ValueError):
            return ExecutionResult(ok=False, intent=req.intent, category=self.category,
                                   action_code=req.action_code,
                                   detail=f"invalid temperature slot {raw!r}",
                                   side_effects=False)
        if not (TEMP_MIN <= temp <= TEMP_MAX):
            return ExecutionResult(ok=False, intent=req.intent, category=self.category,
                                   action_code=req.action_code,
                                   detail=f"temperature {temp} outside {TEMP_MIN}-{TEMP_MAX}",
                                   side_effects=False)
        detail = _spoken_set(temp)
        return ExecutionResult(ok=True, intent=req.intent, category=self.category,
                               action_code=req.action_code,
                               detail=f"hvac: set-point {temp}C (audio-only, no controller wired)",
                               side_effects=False,
                               payload={"temperature": temp, "unit": "C", "answer": detail})
