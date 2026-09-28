"""Music transport (play / pause / stop). Live path: mpg123 / aplay on the Pi."""
from __future__ import annotations

from ..facade import ActionRequest
from .base import ExecutionResult, Executor


class MediaExecutor(Executor):
    category = "media"

    def _live(self, req: ActionRequest) -> ExecutionResult:
        return ExecutionResult(ok=False, intent=req.intent, category=self.category,
                               action_code=req.action_code,
                               detail="media live driver not wired yet",
                               side_effects=False)
