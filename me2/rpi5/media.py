"""Local music playback for the Pi runtime.

Points at a directory of audio files and plays one at a time, chosen at
random (skipping the track that is currently playing / was played last). It
spawns an external player as a child process and drives transport with POSIX
signals, so it works on the demo Pi 5 (Debian, PipeWire) with no extra
services installed.

Backend selection
-----------------
The controller prefers ``ffplay`` (bundled with the ffmpeg package, handles
mp3/flac/ogg/m4a/wav uniformly) and falls back to ``pw-play`` (PipeWire) then
``aplay`` (ALSA). ``ffplay`` alone supports a real pause/resume: pausing sends
SIGSTOP to the child, resuming sends SIGCONT. The fallback players have no
pause, so :meth:`pause` degrades to a stop and :meth:`resume` restarts the
same track.

Demo contract
-------------
Like :class:`~rpi5.volume.VolumeController`, the controller snapshots nothing
global and is fail-soft: a missing player, an empty directory, or a failed
spawn never raises into the mic loop -- it returns a :class:`MediaError` the
executor turns into a spoken apology.

All spawning is routed through an injected ``runner`` callable so the behaviour
is unit-testable without real audio hardware (see ``tests/test_media.py``).
"""
from __future__ import annotations

import logging
import os
import random
import signal
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Sequence

LOGGER = logging.getLogger("pi5-vcm.media")

#: Audio extensions recognised as playable tracks.
AUDIO_EXTS = (".mp3", ".flac", ".ogg", ".oga", ".m4a", ".wav", ".aac", ".opus")

#: Candidate player binaries, in preference order. ``ffplay`` is the only one
#: with real pause/resume (via SIGSTOP/SIGCONT).
PLAYERS = ("ffplay", "pw-play", "aplay")


class MediaError(RuntimeError):
    """Raised when playback cannot start or a transport op is impossible."""


def _pick_player(preferred: str | None = None) -> tuple[str, list[str]]:
    """Return ``(binary, base_args)`` for the best available player.

    ``preferred`` overrides auto-detection (used by tests and by an explicit
    ``--player`` flag). ``base_args`` are the flags that must precede the file
    path for that player.
    """
    if preferred:
        return preferred, _args_for(preferred)
    for name in PLAYERS:
        if shutil.which(name):
            return name, _args_for(name)
    raise MediaError("no audio player found (tried: " + ", ".join(PLAYERS) + ")")


def _args_for(binary: str) -> list[str]:
    if binary == "ffplay":
        return ["-nodisp", "-autoexit", "-loglevel", "quiet"]
    if binary == "pw-play":
        return []
    if binary == "aplay":
        return ["-q"]
    # Unknown binary: hand the file straight to it.
    return []


class MediaPlayerController:
    """Random-playback music transport backed by a spawned player process.

    Parameters
    ----------
    directory:
        Folder scanned (non-recursively) for audio files. May be a
        ``str``/``Path``; expanded (``~``) and resolved at construction.
    player:
        Force a specific player binary (else auto-detected).
    runner:
        Injectable spawner. Signature
        ``runner(cmd: Sequence[str]) -> subprocess.Popen``. Defaults to
        :func:`subprocess.Popen`. Inject a fake in tests.
    rng:
        Injectable :class:`random.Random` (defaults to a fresh instance).
        Inject a seeded one in tests for deterministic picks.
    """

    def __init__(self, directory: str | Path | None = None,
                 *, player: str | None = None,
                 runner: Callable[[Sequence[str]], subprocess.Popen] | None = None,
                 rng: random.Random | None = None) -> None:
        self.directory = (Path(directory).expanduser().resolve()
                          if directory is not None else None)
        self.player = player
        self._runner = runner or subprocess.Popen
        self._rng = rng or random.Random()
        self._proc: subprocess.Popen | None = None
        self._track: Path | None = None
        self._last: Path | None = None
        self._paused = False

    # -- introspection ---------------------------------------------------- #
    @property
    def current_track(self) -> str | None:
        """Filename of the active (or last-started) track, or ``None``."""
        return self._track.name if self._track is not None else None

    @property
    def is_playing(self) -> bool:
        """True when a player process is alive and not paused."""
        return self._proc is not None and self._paused is False and self._alive()

    @property
    def is_paused(self) -> bool:
        return self._proc is not None and self._paused

    def _alive(self) -> bool:
        try:
            return self._proc is not None and self._proc.poll() is None
        except Exception:  # noqa: BLE001
            return False

    # -- discovery -------------------------------------------------------- #
    def discover(self) -> list[Path]:
        """List playable audio files in the directory (sorted, non-recursive)."""
        if self.directory is None or not self.directory.is_dir():
            return []
        files = [p for p in self.directory.iterdir()
                 if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
        return sorted(files)

    def _pick(self) -> Path:
        """Choose a random track, avoiding the last-played one when possible."""
        tracks = self.discover()
        if not tracks:
            raise MediaError(
                f"no audio files found in {self.directory or '(no directory set)'}")
        if len(tracks) > 1 and self._last in tracks:
            pool = [t for t in tracks if t != self._last]
        else:
            pool = tracks
        return self._rng.choice(pool)

    # -- transport -------------------------------------------------------- #
    def play(self, directory: str | Path | None = None,
             before_start: Callable[[str], bool] | None = None) -> str:
        """Resume a paused track, or start a random track otherwise.

        Returns the track filename. Stops any currently running player first,
        except when resuming the paused track.
        ``before_start`` can announce the selected filename before audio begins.
        Raises :class:`MediaError` if the directory is empty or no player is
        available.
        """
        if directory is not None:
            self.directory = Path(directory).expanduser().resolve()
        if self._paused and self._track is not None and directory is None:
            track = self._track
            if before_start is not None:
                before_start(track.name)
            if self.resume():
                return track.name
        self.stop()  # never stack two players
        track = self._pick()
        if before_start is not None:
            before_start(track.name)
        return self._start_track(track)

    def _start_track(self, track: Path) -> str:
        binary, base_args = _pick_player(self.player)
        cmd = [binary, *base_args, str(track)]
        LOGGER.info("[media] play %s via %s", track.name, binary)
        self._proc = self._runner(cmd)
        self._track = track
        self._last = track
        self._paused = False
        return track.name

    def pause(self) -> bool:
        """Pause playback. Returns True if paused, False if nothing to pause.

        With a pausable player this suspends the child (SIGSTOP) so
        :meth:`resume` continues the same track. With a fallback player it
        degrades to a stop.
        """
        if self._proc is None or self._paused or not self._alive():
            return False
        if self.player_supports_pause():
            self._signal(signal.SIGSTOP)
            self._paused = True
            LOGGER.info("[media] paused %s", self.current_track)
        else:
            # No pause on this backend: stop and remember the track.
            self._terminate()
            self._paused = True
            LOGGER.info("[media] stopped (no pause on %s) %s",
                        self.player or "?", self.current_track)
        return True

    def resume(self) -> bool:
        """Resume a paused track. Returns True if resumed, else False."""
        if self._proc is None or not self._paused:
            return False
        if self.player_supports_pause() and self._alive():
            self._signal(signal.SIGCONT)
            self._paused = False
            LOGGER.info("[media] resumed %s", self.current_track)
            return True
        # Fallback player: restart the remembered track from the top.
        track = self._track
        if track is None:
            return False
        self._proc = None
        self._paused = False
        self._start_track(track)
        return True

    def stop(self) -> bool:
        """Stop playback. Returns True if something was stopped."""
        if self._proc is None:
            self._track = None
            self._paused = False
            return False
        self._terminate()
        self._proc = None
        self._track = None
        self._paused = False
        LOGGER.info("[media] stopped")
        return True

    # -- helpers ---------------------------------------------------------- #
    def player_supports_pause(self) -> bool:
        """True when the active backend can suspend in place (ffplay)."""
        binary = self.player or (shutil.which("ffplay") and "ffplay"
                                 or (shutil.which("pw-play") and "pw-play")
                                 or (shutil.which("aplay") and "aplay"))
        return binary == "ffplay"

    def _signal(self, sig: int) -> None:
        try:
            self._proc.send_signal(sig)
        except (ProcessLookupError, OSError) as exc:
            LOGGER.warning("[media] signal %s failed: %s", sig, exc)

    def _terminate(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            # If suspended, unfreeze first so it can exit cleanly.
            if self._paused:
                proc.send_signal(signal.SIGCONT)
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        except (ProcessLookupError, OSError):
            pass
        finally:
            self._paused = False
