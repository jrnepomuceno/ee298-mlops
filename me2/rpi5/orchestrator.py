"""Orchestrator: route an ActionRequest to its category executor and drive I/O.

Sits between the facade (which decides *what*) and the hardware (which does
*it*). Responsibilities:

* Route by :attr:`ActionRequest.category` to the right executor.
* Drive the RGB state machine around execution (processing -> speaking -> idle).
* Speak the reply after the action resolves.
* Emit one structured JSON event per utterance (for logging / the dashboard).
* Fail soft: an executor error becomes a spoken apology + error RGB, never a
  crash in the mic loop.

The orchestrator is deliberately synchronous. Slow executors (weather, calls)
are isolated by giving them their own thread in the caller if needed; the
default executors are all fast/local, so a straight call keeps the code simple
and the Pi's single-core budget predictable.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping
from uuid import uuid4

from .facade import ActionRequest, RejectResult
from .executors.base import ExecutionResult, Executor

LOGGER = logging.getLogger("pi5-vcm.orchestrator")


@dataclass(frozen=True)
class OrchestratorResult:
    """Full outcome of one utterance through the facade + orchestrator."""

    handled: bool                 # True if an action ran (even dry-run)
    request: ActionRequest | None
    execution: ExecutionResult | None
    reject: RejectResult | None
    reply_text: str
    event: Mapping[str, Any] = field(default_factory=dict)


class Orchestrator:
    def __init__(
        self,
        executors: Mapping[str, Executor] | None = None,
        *,
        rgb: Any | None = None,
        speak: Callable[[str], None] | None = None,
        on_event: Callable[[Mapping[str, Any]], None] | None = None,
        default_executors: Mapping[str, Executor] | None = None,
    ) -> None:
        self.rgb = rgb
        self.speak = speak
        self.on_event = on_event
        self._executors: dict[str, Executor] = dict(default_executors or {})
        for cat, ex in (executors or {}).items():
            self._executors[cat] = ex

    def register(self, category: str, executor: Executor) -> None:
        self._executors[category] = executor

    # -- RGB helpers (tolerate a missing controller) ---------------------- #
    def _rgb(self, *methods: str) -> None:
        if self.rgb is None:
            return
        for m in methods:
            fn = getattr(self.rgb, m, None)
            if callable(fn):
                fn()

    # -- Main entry point ------------------------------------------------- #
    def run(self, request_or_reject: ActionRequest | RejectResult,
            source: str = "microphone") -> OrchestratorResult:
        if isinstance(request_or_reject, RejectResult):
            return self._handle_reject(request_or_reject, source)
        return self._handle_action(request_or_reject, source)

    def _handle_reject(self, rej: RejectResult, source: str) -> OrchestratorResult:
        self._rgb("processing")
        self._speak(rej.reply_text)
        self._rgb("idle")
        event = self._event(source, intent=rej.intent or "rejected",
                            status="rejected", code=rej.reason, detail=rej.detail,
                            reply=rej.reply_text)
        return OrchestratorResult(handled=False, request=None, execution=None,
                                  reject=rej, reply_text=rej.reply_text, event=event)

    def _handle_action(self, req: ActionRequest, source: str) -> OrchestratorResult:
        self._rgb("processing")
        executor = self._executors.get(req.category)
        if executor is None:
            err = ExecutionResult(ok=False, intent=req.intent, category=req.category,
                                  action_code=req.action_code,
                                  detail=f"no executor for category '{req.category}'",
                                  side_effects=False)
            reply = "I can't do that yet."
        else:
            try:
                err = executor.run(req)
            except Exception as exc:  # noqa: BLE001 - fail soft, never crash the loop
                LOGGER.exception("executor %s raised", req.category)
                err = ExecutionResult(ok=False, intent=req.intent, category=req.category,
                                      action_code=req.action_code,
                                      detail=f"executor error: {exc}", side_effects=False)
            # Query-style intents (time, weather, reminders) put the
            # authoritative value in payload["answer"]; the executor is the
            # single source of truth, so prefer it over the facade's static
            # template. Command intents carry no "answer", so they keep the
            # template.
            if err.ok:
                answer = (err.payload or {}).get("answer")
                reply = str(answer) if answer else req.reply_text
            else:
                reply = "Sorry, that didn't work."
        self._speak(reply)
        self._rgb("speaking", "idle")
        event = self._event(source, intent=req.intent, status=("acted" if err.ok else "error"),
                            code=req.action_code, detail=err.detail, reply=reply,
                            category=req.category, slots=dict(req.slots),
                            side_effects=err.side_effects)
        return OrchestratorResult(handled=err.ok, request=req, execution=err,
                                  reject=None, reply_text=reply, event=event)

    def _speak(self, text: str) -> None:
        if self.speak is not None and text:
            try:
                self.speak(text)
            except Exception:  # noqa: BLE001
                LOGGER.exception("speak() failed")

    def _event(self, source: str, **fields: Any) -> Mapping[str, Any]:
        ev: dict[str, Any] = {
            "id": uuid4().hex,
            "ts": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "pipeline": "facade+orchestrator",
        }
        ev.update(fields)
        if self.on_event is not None:
            try:
                self.on_event(ev)
            except Exception:  # noqa: BLE001
                LOGGER.exception("on_event() failed")
        LOGGER.debug("event %s", json.dumps(ev, default=str))
        return ev


def default_orchestrator(*, rgb: Any | None = None,
                         speak: Callable[[str], None] | None = None,
                         on_event: Callable[[Mapping[str, Any]], None] | None = None,
                         dry_run: bool = True,
                         weather_fn: Callable[[], str] | None = None,
                         timer_manager: Any | None = None,
                         reminder_store: Any | None = None,
                         volume_controller: Any | None = None,
                         media_player: Any | None = None,
                         media_volume: int | None = None,
                         light_driver: Any | None = None,
                         dialer: Any | None = None) -> Orchestrator:
    """Build an orchestrator wired to the stock dry-run executors for all 7 categories.

    ``weather_fn`` (optional) is passed to :class:`InfoExecutor` so the
    ``what_weather`` intent can return a live, spoken-friendly line.
    ``timer_manager`` (optional) is passed to :class:`TimerExecutor` so the
    timer intents can schedule / cancel a real in-process timer.
    ``reminder_store`` (optional) is passed to :class:`ReminderExecutor` and
    :class:`InfoExecutor` so reminders can be persisted and read back.
    ``volume_controller`` (optional) is passed to :class:`VolumeExecutor` so
    the volume intents can move the real system output volume.
    ``media_player`` (optional) is passed to :class:`MediaExecutor` so the
    media intents (play/pause/stop) can drive real local playback.
    ``light_driver`` (optional) is passed to :class:`LightExecutor` so the
    light intents (on/off/dim) can drive a real device (default target: the
    HyperX DuoCast ring light).
    ``dialer`` (optional) is passed to :class:`CommsExecutor` so the
    ``call`` intent can place a real SIP call (default target: baresip).
    """
    from .executors import (
        LightExecutor, HvacExecutor, MediaExecutor, TimerExecutor,
        ReminderExecutor, CommsExecutor, InfoExecutor, VolumeExecutor,
    )
    ex = {
        "lights": LightExecutor(dry_run, driver=light_driver),
        "hvac": HvacExecutor(dry_run),
        "media": MediaExecutor(dry_run, player=media_player,
                              volume_controller=volume_controller,
                              media_volume=media_volume),
        "timer": TimerExecutor(dry_run, manager=timer_manager),
        "remind": ReminderExecutor(dry_run, store=reminder_store),
        "comms": CommsExecutor(dry_run, dialer=dialer),
        "info": InfoExecutor(dry_run, weather_fn=weather_fn,
                             reminders_fn=(reminder_store.summarize
                                           if reminder_store is not None else None)),
        "volume": VolumeExecutor(dry_run, controller=volume_controller),
    }
    return Orchestrator(ex, rgb=rgb, speak=speak, on_event=on_event)
