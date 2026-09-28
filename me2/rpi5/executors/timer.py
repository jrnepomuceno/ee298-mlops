"""Timers & alarms. Live path: the local TimerManager (already in the harness)."""
from __future__ import annotations

from ..facade import ActionRequest
from ..timer import SECONDS_PER_UNIT
from .base import ExecutionResult, Executor


def _spoken_set(duration: int, unit: str) -> str:
    """Spoken confirmation for a countdown timer, TTS-friendly.

    Singular/plural is derived from the value; the unit is normalized to its
    base word so "30 second(s)" / "5 minute(s)" / "1 hour" all read cleanly.
    """
    u = str(unit).rstrip("s").lower()
    if u.startswith("sec"):
        unit_word = "second"
    elif u.startswith("hour"):
        unit_word = "hour"
    else:
        unit_word = "minute"
    plural = "" if int(duration) == 1 else "s"
    return f"Timer set for {int(duration)} {unit_word}{plural}."


class TimerExecutor(Executor):
    """Wraps the harness :class:`~rpi5.timer.TimerManager` when provided.

    In dry-run it behaves like the base (no side effects). With a ``manager``
    and ``dry_run=False`` it actually schedules / cancels timers. The unit is
    honored (second/minute/hour) -- it is never silently treated as minutes.
    """

    category = "timer"

    def __init__(self, dry_run: bool = True, manager: object | None = None) -> None:
        super().__init__(dry_run)
        self.manager = manager

    def _live(self, req: ActionRequest) -> ExecutionResult:
        mgr = self.manager
        if mgr is None:
            return ExecutionResult(ok=False, intent=req.intent, category=self.category,
                                   action_code=req.action_code,
                                   detail="no TimerManager attached", side_effects=False)
        try:
            if req.action_code == "timer.set":
                duration = int(req.slots.get("duration", 0))
                unit = str(req.slots.get("duration_unit", "minute"))
                if duration <= 0:
                    return ExecutionResult(ok=False, intent=req.intent, category=self.category,
                                           action_code=req.action_code,
                                           detail=f"invalid timer duration {duration}",
                                           side_effects=False)
                if unit.rstrip("s").lower() not in SECONDS_PER_UNIT:
                    return ExecutionResult(ok=False, intent=req.intent, category=self.category,
                                           action_code=req.action_code,
                                           detail=f"unsupported timer unit {unit!r}",
                                           side_effects=False)
                mgr.set_timer(duration, unit)
                return ExecutionResult(ok=True, intent=req.intent, category=self.category,
                                       action_code=req.action_code,
                                       detail=f"timer live: {req.action_code}",
                                       side_effects=True,
                                       payload={"answer": _spoken_set(duration, unit)})
            elif req.action_code == "alarm.set":
                # Wall-clock alarm: not wired yet. Acknowledge gracefully (like the
                # weather fallback) rather than failing soft, so the user hears the
                # honest reason instead of a generic apology.
                return ExecutionResult(ok=True, intent=req.intent, category=self.category,
                                       action_code=req.action_code,
                                       detail="alarms are not wired yet",
                                       side_effects=False,
                                       payload={"answer": "Alarms aren't available yet."})
            elif req.action_code == "timer.cancel":
                res = mgr.cancel()
                cancelled = res.get("side_effects")
                answer = ("Timer stopped." if cancelled
                          else "There's no timer running.")
                return ExecutionResult(ok=True, intent=req.intent, category=self.category,
                                       action_code=req.action_code,
                                       detail=f"timer live: {req.action_code}",
                                       side_effects=bool(cancelled),
                                       payload={"answer": answer})
            else:
                return ExecutionResult(ok=False, intent=req.intent, category=self.category,
                                       action_code=req.action_code,
                                       detail=f"unsupported timer action {req.action_code}",
                                       side_effects=False)
        except Exception as exc:  # noqa: BLE001 - surface any driver error as a result
            return ExecutionResult(ok=False, intent=req.intent, category=self.category,
                                   action_code=req.action_code,
                                   detail=f"timer error: {exc}", side_effects=False)
