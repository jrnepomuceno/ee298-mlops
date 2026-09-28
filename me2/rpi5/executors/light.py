"""Light control (turn on / off / dim). Live path: GPIO PWM or a smart-home API."""
from __future__ import annotations

from ..facade import ActionRequest
from .base import ExecutionResult, Executor


class LightExecutor(Executor):
    category = "lights"

    def _live(self, req: ActionRequest) -> ExecutionResult:
        # Placeholder for GPIO PWM / MQTT / Home Assistant. Kept explicit so a
        # live wiring is a single, reviewable edit.
        return ExecutionResult(ok=False, intent=req.intent, category=self.category,
                               action_code=req.action_code,
                               detail="lights live driver not wired yet",
                               side_effects=False)
