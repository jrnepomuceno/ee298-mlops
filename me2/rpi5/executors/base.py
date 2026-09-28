"""Executor contract shared by all category executors."""
from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..facade import ActionRequest

LOGGER = logging.getLogger("pi5-vcm.executor")


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of executing one :class:`ActionRequest`."""

    ok: bool
    intent: str
    category: str
    action_code: str
    detail: str = ""
    side_effects: bool = False     # True only when real hardware/network was touched
    payload: Mapping[str, Any] = field(default_factory=dict)


class Executor(abc.ABC):
    """Base for category executors.

    Subclasses set :attr:`category` and implement :meth:`run`. The default
    :meth:`run` is a dry-run: it logs the request and reports success with no
    side effects. Live executors override :meth:`run` and call
    :meth:`_live` (which subclasses fill in) when ``dry_run`` is False.
    """

    category: str = "base"

    def __init__(self, dry_run: bool = True) -> None:
        self.dry_run = dry_run

    def run(self, req: ActionRequest) -> ExecutionResult:
        if not self.dry_run:
            return self._live(req)
        LOGGER.info("[dry-run] %s %s slots=%s", req.action_code, req.intent, dict(req.slots))
        return ExecutionResult(
            ok=True, intent=req.intent, category=self.category,
            action_code=req.action_code,
            detail=f"dry-run: {req.action_code} accepted",
            side_effects=False,
        )

    def _live(self, req: ActionRequest) -> ExecutionResult:
        raise NotImplementedError(f"{self.category} has no live implementation yet")
