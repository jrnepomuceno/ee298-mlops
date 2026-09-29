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
            # %-I is glibc-only; on musl (some Pi images) it raises ValueError.
            # Compute the 12-hour hour manually so the answer is identical
            # everywhere.
            now = datetime.now()
            clock = f"{now.hour % 12 or 12}:{now.strftime('%M')} {now.strftime('%p')}"
            # The spoken line must read as a full sentence ("The time is 1:45
            # PM."), not a bare clock value. The orchestrator prefers
            # payload["answer"] over the facade template, so the sentence has
            # to live here -- otherwise the "The time is" wrapper is dropped
            # and the assistant only says the time.
            answer = f"The time is {clock}."
            detail = f"time is {clock}"
        elif req.action_code == "query.weather":
            # A configured provider is the gate, not dry_run: the weather line
            # is read-only (no hardware side effect), so it is safe to fetch
            # even when the pipeline runs dry-run. dry_run only suppresses
            # *actuating* intents (lights, calls, volume, ...), never an answer.
            if self.weather_fn is not None:
                try:
                    answer = self.weather_fn()
                except Exception:  # noqa: BLE001
                    # A failed lookup is usually a missing API key, no network,
                    # or the service being unreachable. Speak a calm apology
                    # instead of echoing the raw exception (which reads as
                    # "Weather lookup failed: <urlopen error ...>").
                    answer = ("I'm sorry, I couldn't reach the weather "
                              "service right now.")
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
