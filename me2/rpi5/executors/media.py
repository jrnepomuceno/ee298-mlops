"""Music transport (play / pause / stop).

Live path: :class:`~rpi5.media.MediaPlayerController`, which points at a
directory of audio files and spawns an external player (``ffplay`` preferred,
``pw-play``/``aplay`` fallback) to play a random track. Like the other
executors it ships in ``dry_run=True`` mode by default: it validates and
reports the action it *would* take without touching audio. With a real
:class:`MediaPlayerController` attached and ``dry_run=False`` it drives real
playback on the Pi.

The controller is constructed in ``run.py`` (``--music-dir`` / ``--live-media``)
and threaded through :func:`~rpi5.orchestrator.default_orchestrator` ->
:class:`~rpi5.harness.FacadePipeline`.
"""
from __future__ import annotations

from ..facade import ActionRequest
from ..media import MediaPlayerController
from .base import ExecutionResult, Executor


class MediaExecutor(Executor):
    category = "media"

    def __init__(self, dry_run: bool = True,
                 player: MediaPlayerController | None = None) -> None:
        super().__init__(dry_run)
        self.player = player

    def _live(self, req: ActionRequest) -> ExecutionResult:
        ctrl = self.player
        if ctrl is None:
            return ExecutionResult(
                ok=False, intent=req.intent, category=self.category,
                action_code=req.action_code,
                detail="no media player attached", side_effects=False)
        code = req.action_code
        try:
            if code == "media.play":
                name = ctrl.play()
                detail = f"Playing {name}"
            elif code == "media.pause":
                if ctrl.pause():
                    detail = f"Paused {ctrl.current_track}"
                else:
                    detail = "Nothing is playing"
            elif code == "media.stop":
                if ctrl.stop():
                    detail = "Stopped the music"
                else:
                    detail = "Nothing is playing"
            else:
                return ExecutionResult(
                    ok=False, intent=req.intent, category=self.category,
                    action_code=code,
                    detail=f"unsupported media action {code}",
                    side_effects=False)
            return ExecutionResult(
                ok=True, intent=req.intent, category=self.category,
                action_code=code, detail=detail, side_effects=True,
                payload={"track": ctrl.current_track, "answer": detail})
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                ok=False, intent=req.intent, category=self.category,
                action_code=code, detail=f"music playback failed: {exc}",
                side_effects=False)
