"""Thermostat / temperature set. Live path: HVAC controller or GPIO relay."""
from __future__ import annotations

from ..facade import ActionRequest
from .base import ExecutionResult, Executor


class HvacExecutor(Executor):
    category = "hvac"

    def _live(self, req: ActionRequest) -> ExecutionResult:
        return ExecutionResult(ok=False, intent=req.intent, category=self.category,
                               action_code=req.action_code,
                               detail="hvac live driver not wired yet",
                               side_effects=False)
