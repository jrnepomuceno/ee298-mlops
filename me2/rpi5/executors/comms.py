"""Outbound calls. Live path: Twilio REST (cloud is allowed for comms)."""
from __future__ import annotations

from ..facade import ActionRequest
from .base import ExecutionResult, Executor


class CommsExecutor(Executor):
    category = "comms"

    def __init__(self, dry_run: bool = True, dialer: object | None = None) -> None:
        super().__init__(dry_run)
        self.dialer = dialer  # optional object with .call(contact)

    def _live(self, req: ActionRequest) -> ExecutionResult:
        dialer = self.dialer
        if dialer is None:
            return ExecutionResult(ok=False, intent=req.intent, category=self.category,
                                   action_code=req.action_code,
                                   detail="no dialer attached", side_effects=False)
        try:
            dialer.call(str(req.slots.get("contact", "")))
            return ExecutionResult(ok=True, intent=req.intent, category=self.category,
                                   action_code=req.action_code,
                                   detail="call placed", side_effects=True)
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(ok=False, intent=req.intent, category=self.category,
                                   action_code=req.action_code,
                                   detail=f"call error: {exc}", side_effects=False)
