"""Reminders. Live path: local JSON persistence (offline, no network)."""
from __future__ import annotations

from ..facade import ActionRequest
from .base import ExecutionResult, Executor


class ReminderExecutor(Executor):
    category = "remind"

    def __init__(self, dry_run: bool = True, store: object | None = None) -> None:
        super().__init__(dry_run)
        self.store = store  # optional object with .add(note) / .list()

    def _live(self, req: ActionRequest) -> ExecutionResult:
        store = self.store
        if store is None:
            return ExecutionResult(ok=False, intent=req.intent, category=self.category,
                                   action_code=req.action_code,
                                   detail="no reminder store attached", side_effects=False)
        try:
            store.add(str(req.slots.get("note", "")))
            return ExecutionResult(ok=True, intent=req.intent, category=self.category,
                                   action_code=req.action_code,
                                   detail="reminder saved", side_effects=True)
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(ok=False, intent=req.intent, category=self.category,
                                   action_code=req.action_code,
                                   detail=f"reminder error: {exc}", side_effects=False)
