"""Information queries (time, weather, reminders). Weather may use the network."""
from __future__ import annotations

from datetime import datetime
from typing import Callable

from ..facade import ActionRequest
from .base import ExecutionResult, Executor


class InfoExecutor(Executor):
    """Answers local queries immediately; delegates weather to a provider.

    ``weather_fn`` is an optional callable ``() -> str`` returning a spoken
    weather line. Without it (or in dry-run) a neutral line is returned.
    """

    category = "info"

    def __init__(self, dry_run: bool = True,
                 weather_fn: Callable[[], str] | None = None,
                 reminders_fn: Callable[[], str] | None = None) -> None:
        super().__init__(dry_run)
        self.weather_fn = weather_fn
        self.reminders_fn = reminders_fn

    def run(self, req: ActionRequest) -> ExecutionResult:
        # Info queries always produce a spoken answer, even in dry-run, because
        # the "action" IS the answer (the time, the weather line).
        if req.action_code == "query.time":
            answer = datetime.now().strftime("%-I:%M %p")
            detail = f"time is {answer}"
        elif req.action_code == "query.weather":
            # A configured provider is the gate, not dry_run: the weather line
            # is read-only (no hardware side effect), so it is safe to fetch
            # even when the pipeline runs dry-run. dry_run only suppresses
            # *actuating* intents (lights, calls, volume, ...), never an answer.
            if self.weather_fn is not None:
                try:
                    answer = self.weather_fn()
                except Exception as exc:  # noqa: BLE001
                    answer = f"Weather lookup failed: {exc}"
            else:
                answer = "Weather is not available without a configured source."
            detail = answer
        elif req.action_code == "query.reminders":
            # Reading reminders is a local, offline operation (no network), so
            # we answer from the store even in dry-run -- the "action" IS the
            # list. Without a store attached we fall back to a neutral line.
            if self.reminders_fn is not None:
                try:
                    answer = self.reminders_fn()
                except Exception as exc:  # noqa: BLE001
                    answer = f"Could not read reminders: {exc}"
            else:
                answer = "You have no reminders."
            detail = answer
        else:
            return ExecutionResult(ok=False, intent=req.intent, category=self.category,
                                   action_code=req.action_code,
                                   detail=f"unsupported info action {req.action_code}",
                                   side_effects=False)
        return ExecutionResult(ok=True, intent=req.intent, category=self.category,
                               action_code=req.action_code, detail=detail,
                               side_effects=False, payload={"answer": answer})
