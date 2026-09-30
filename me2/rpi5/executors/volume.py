"""Output-volume control (volume up / down / mute).

Live path: :class:`~rpi5.volume.VolumeController`, which drives the overall
volume of the connected output speaker driver. Like the other executors it
ships in ``dry_run=True`` mode by default: it validates and reports the level
it *would* set without touching hardware. With a real :class:`VolumeController`
attached and ``dry_run=False`` it moves the actual system volume.

The controller is shared with the demo lifecycle (``run.py``) so the pre-demo
level is captured once at start and restored once at shutdown -- the executor
only ever steps / mutes, it never owns the restore.
"""
from __future__ import annotations

from ..facade import ActionRequest
from ..volume import STEP_PERCENT, VolumeController
from .base import ExecutionResult, Executor


class VolumeExecutor(Executor):
    category = "volume"

    def __init__(self, dry_run: bool = True,
                 controller: VolumeController | None = None) -> None:
        super().__init__(dry_run)
        self.controller = controller

    def _live(self, req: ActionRequest) -> ExecutionResult:
        ctrl = self.controller
        if ctrl is None:
            return ExecutionResult(
                ok=False, intent=req.intent, category=self.category,
                action_code=req.action_code,
                detail="no volume controller attached", side_effects=False)
        code = req.action_code
        try:
            if code == "volume.up":
                level = ctrl.step(STEP_PERCENT)
                detail = f"volume up to {level}%"
            elif code == "volume.down":
                level = ctrl.step(-STEP_PERCENT)
                detail = f"volume down to {level}%"
            elif code == "volume.unmute":
                level = ctrl.unmute()
                detail = f"unmuted to {level}%"
            else:
                return ExecutionResult(
                    ok=False, intent=req.intent, category=self.category,
                    action_code=code,
                    detail=f"unsupported volume action {code}",
                    side_effects=False)
            return ExecutionResult(
                ok=True, intent=req.intent, category=self.category,
                action_code=code, detail=detail, side_effects=True,
                payload={"level": level, "answer": detail})
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                ok=False, intent=req.intent, category=self.category,
                action_code=code, detail=f"volume control failed: {exc}",
                side_effects=False)
