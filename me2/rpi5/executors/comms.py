"""Outbound calls.

Live path: a pluggable :class:`~rpi5.calls.Dialer` selected at startup
(``--dialer NAME``). The default target is :program:`baresip`
(:class:`~rpi5.calls.BaresipDialer`), but the executor only speaks the generic
dialer interface, so swapping the dial device behind the ``call`` intent is a
flag change, never an edit here.

Like the other executors it ships in ``dry_run=True`` mode by default: it
validates the contact and reports the call it *would* place without touching the
network. With a real dialer attached and ``dry_run=False`` it places the actual
call.

The dialer is constructed in ``run.py`` (``--dialer`` / ``--baresip-config`` /
``--baresip-account``) and threaded through
:func:`~rpi5.orchestrator.default_orchestrator` ->
:class:`~rpi5.harness.FacadePipeline`.
"""
from __future__ import annotations

from ..calls import Dialer, normalize_target
from ..facade import ActionRequest
from .base import ExecutionResult, Executor


class CommsExecutor(Executor):
    category = "comms"

    def __init__(self, dry_run: bool = True,
                 dialer: Dialer | None = None) -> None:
        super().__init__(dry_run)
        self.dialer = dialer

    def _live(self, req: ActionRequest) -> ExecutionResult:
        dialer = self.dialer
        if dialer is None:
            return ExecutionResult(
                ok=False, intent=req.intent, category=self.category,
                action_code=req.action_code,
                detail="no dialer attached", side_effects=False)
        code = req.action_code
        try:
            if code == "call.request":
                contact = str(req.slots.get("contact", ""))
                target = normalize_target(contact)
                dialer.dial(target)
                detail = f"Calling {contact}."
                payload = {"contact": contact, "target": target,
                           "dialer": dialer.name, "answer": detail}
                return ExecutionResult(
                    ok=True, intent=req.intent, category=self.category,
                    action_code=code, detail=detail, side_effects=True,
                    payload=payload)
            if code == "call.cancel":
                dialer.cancel()
                detail = "Ending the call."
                return ExecutionResult(
                    ok=True, intent=req.intent, category=self.category,
                    action_code=code, detail=detail, side_effects=True,
                    payload={"dialer": dialer.name, "answer": detail})
            return ExecutionResult(
                ok=False, intent=req.intent, category=self.category,
                action_code=code,
                detail=f"unsupported comms action {code}", side_effects=False)
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                ok=False, intent=req.intent, category=self.category,
                action_code=code, detail=f"call failed: {exc}",
                side_effects=False)
