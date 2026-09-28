"""Light control (turn on / off / dim).

Live path: a pluggable :class:`~rpi5.rgb.LightDriver` selected at startup
(``--light-driver NAME``). The default target is the HyperX DuoCast
microphone ring light (:class:`~rpi5.rgb.HyperxDuoCastDriver`), but the
executor only speaks the generic driver interface, so swapping the device
behind the voice commands is a flag change, never an edit here.

Like the other executors it ships in ``dry_run=True`` mode by default: it
validates and reports the action it *would* take without touching hardware.
With a real driver attached and ``dry_run=False`` it drives the actual light.

The driver is constructed in ``run.py`` (``--light-driver`` /
``--light-executable``) and threaded through
:func:`~rpi5.orchestrator.default_orchestrator` ->
:class:`~rpi5.harness.FacadePipeline`.
"""
from __future__ import annotations

from ..facade import ActionRequest
from ..rgb import LightDriver
from .base import ExecutionResult, Executor


class LightExecutor(Executor):
    category = "lights"

    def __init__(self, dry_run: bool = True,
                 driver: LightDriver | None = None) -> None:
        super().__init__(dry_run)
        self.driver = driver

    def _live(self, req: ActionRequest) -> ExecutionResult:
        drv = self.driver
        if drv is None:
            return ExecutionResult(
                ok=False, intent=req.intent, category=self.category,
                action_code=req.action_code,
                detail="no light driver attached", side_effects=False)
        code = req.action_code
        try:
            if code == "lights.on":
                drv.on()
                # LightDriver.on() restores the driver's last brightness; read
                # it back for the payload (drivers expose it as _brightness).
                level = getattr(drv, "_brightness", None)
                detail = "Turning the lights on."
            elif code == "lights.off":
                drv.off()
                level = 0
                detail = "Turning the lights off."
            elif code == "lights.dim":
                percent = int(req.slots["percent"])
                level = drv.set_brightness(percent)
                detail = f"Dimming the lights to {level} percent."
            else:
                return ExecutionResult(
                    ok=False, intent=req.intent, category=self.category,
                    action_code=code,
                    detail=f"unsupported light action {code}",
                    side_effects=False)
            return ExecutionResult(
                ok=True, intent=req.intent, category=self.category,
                action_code=code, detail=detail, side_effects=True,
                payload={"driver": drv.name, "level": level, "answer": detail})
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                ok=False, intent=req.intent, category=self.category,
                action_code=code, detail=f"light control failed: {exc}",
                side_effects=False)
