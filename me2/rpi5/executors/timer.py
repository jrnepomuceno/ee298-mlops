"""Timers & alarms. Live path: the local TimerManager (already in the harness)."""
from __future__ import annotations

from ..facade import ActionRequest
from .base import ExecutionResult, Executor


class TimerExecutor(Executor):
    """Wraps the harness :class:`~rpi5.timer.TimerManager` when provided.

    In dry-run it behaves like the base. With a ``manager`` and ``dry_run=False``
    it actually schedules / cancels timers.
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
                secs = int(req.slots.get("duration", 0))
                unit = str(req.slots.get("duration_unit", "minute"))
                if unit.startswith("min"):
                    secs *= 60
                mgr.set_timer(secs)
            elif req.action_code == "alarm.set":
                mgr.set_alarm(str(req.slots.get("time", "")))
            elif req.action_code == "timer.cancel":
                mgr.cancel()
            else:
                return ExecutionResult(ok=False, intent=req.intent, category=self.category,
                                       action_code=req.action_code,
                                       detail=f"unsupported timer action {req.action_code}",
                                       side_effects=False)
            return ExecutionResult(ok=True, intent=req.intent, category=self.category,
                                   action_code=req.action_code,
                                   detail=f"timer live: {req.action_code}",
                                   side_effects=True)
        except Exception as exc:  # noqa: BLE001 - surface any driver error as a result
            return ExecutionResult(ok=False, intent=req.intent, category=self.category,
                                   action_code=req.action_code,
                                   detail=f"timer error: {exc}", side_effects=False)
