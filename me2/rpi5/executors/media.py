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

from pathlib import Path
from typing import Callable

from ..facade import ActionRequest
from ..media import MediaError, MediaPlayerController
from .base import ExecutionResult, Executor


class MediaExecutor(Executor):
    category = "media"

    def __init__(self, dry_run: bool = True,
                 player: MediaPlayerController | None = None,
                 volume_controller=None,
                 media_volume: int | None = None,
                 before_play: Callable[[str], bool] | None = None) -> None:
        super().__init__(dry_run)
        self.player = player
        self.volume_controller = volume_controller
        self.media_volume = media_volume
        self.before_play = before_play

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
                announced_before_play = False
                was_paused = ctrl.is_paused

                def announce_before_start(track_name: str) -> bool:
                    nonlocal announced_before_play
                    if self.before_play is not None:
                        announced_before_play = bool(self.before_play(track_name))
                    return announced_before_play

                name = ctrl.play(
                    before_start=(announce_before_start
                                  if self.before_play is not None else None))
                self._apply_media_volume()
                verb = "Resuming" if was_paused else "Playing"
                detail = f"{verb} {Path(name).stem}"
            elif code == "media.pause":
                if ctrl.pause():
                    detail = "Music paused"
                else:
                    detail = "No music is playing."
            elif code == "media.stop":
                if ctrl.stop():
                    detail = "Music stopped"
                else:
                    detail = "No music is playing."
            else:
                return ExecutionResult(
                    ok=False, intent=req.intent, category=self.category,
                    action_code=code,
                    detail=f"unsupported media action {code}",
                    side_effects=False)
            payload = {"track": ctrl.current_track, "answer": detail}
            if code == "media.play" and announced_before_play:
                payload["announced_before_play"] = True
            return ExecutionResult(
                ok=True, intent=req.intent, category=self.category,
                action_code=code, detail=detail, side_effects=True,
                payload=payload)
        except MediaError:
            # No music available (empty directory or no player binary). This is
            # a normal, expected situation -- not an error -- so the assistant
            # acknowledges the command and plays the canned "play music" reply
            # instead of falling through to the generic "Sorry, that didn't
            # work." apology.
            return ExecutionResult(
                ok=True, intent=req.intent, category=self.category,
                action_code=code,
                detail="I couldn't find any music to play right now.",
                side_effects=False,
                payload={"track": None,
                         "answer": "I couldn't find any music to play right now."})
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                ok=False, intent=req.intent, category=self.category,
                action_code=code, detail=f"music playback failed: {exc}",
                side_effects=False)

    def _apply_media_volume(self) -> None:
        """Set the media stream's own PipeWire volume, if configured.

        Keeps music at ``media_volume`` independently of the master sink that
        TTS/alarms use. Fail-soft: no controller, no level, or a backend
        without per-stream support (``set_channel`` returns ``None``) simply
        leaves the stream at whatever level PipeWire assigned it.
        """
        if (self.volume_controller is None
                or self.media_volume is None
                or self.player is None):
            return
        try:
            channel = getattr(self.player, "player", None) or "ffplay"
            self.volume_controller.set_channel(channel, self.media_volume)
        except Exception:  # noqa: BLE001
            pass
