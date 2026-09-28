"""System output-volume control for the Pi runtime.

Drives the *overall* volume of the connected output speaker driver. It is
platform-aware so the same code works on the demo Pi 5 (PipeWire ``wpctl``
with an ALSA ``amixer`` fallback) and on a developer Mac (CoreAudio via
``osascript``).

Demo contract
-------------
The controller snapshots the current output volume the moment the demo starts
(:meth:`VolumeController.begin`) and restores it the moment the demo ends
(:meth:`VolumeController.end`). Restoring is idempotent and fail-soft: if the
volume was never captured, or the restore command fails, the process keeps
running -- a volume hiccup must never take the mic loop down.

All commands are injected as callables so the behaviour is unit-testable
without touching real audio hardware (see ``tests/test_volume.py``).
"""
from __future__ import annotations

import logging
import platform
import re
import shutil
import subprocess
from typing import Callable

LOGGER = logging.getLogger("pi5-vcm.volume")

#: How much the master volume moves on each "louder" / "softer" command.
STEP_PERCENT = 10
MIN_LEVEL = 0
MAX_LEVEL = 100


def _clamp(level: int) -> int:
    return max(MIN_LEVEL, min(MAX_LEVEL, int(level)))


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    """Run a command, return stripped stdout. Raises on non-zero exit."""
    proc = subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout, text=True,
    )
    return proc.stdout.strip()


def _detect_backend() -> str:
    """Pick a backend for this host: ``wpctl`` | ``amixer`` | ``osascript``."""
    system = platform.system().lower()
    if system == "darwin":
        return "osascript"
    if shutil.which("wpctl"):
        return "wpctl"
    if shutil.which("amixer"):
        return "amixer"
    return "none"


class VolumeController:
    """Capture / adjust / restore the system output volume.

    Parameters
    ----------
    backend:
        Force a backend (``wpctl``/``amixer``/``osascript``/``none``).
        Defaults to auto-detection.
    get/set:
        Optional overrides ``fn() -> int`` and ``fn(level:int) -> None``.
        When supplied, the real subprocess commands are bypassed entirely --
        this is what the unit tests use.
    """

    def __init__(self, backend: str | None = None,
                 get: Callable[[], int] | None = None,
                 set: Callable[[int], None] | None = None) -> None:
        self.backend = backend or _detect_backend()
        self._get_override = get
        self._set_override = set
        self._captured: int | None = None
        self._restored = False

    # ------------------------------------------------------------------ #
    # Backend plumbing (real hardware)
    # ------------------------------------------------------------------ #
    def _get_level(self) -> int:
        if self._get_override is not None:
            return _clamp(self._get_override())
        if self.backend == "osascript":
            out = _run(["osascript", "-e",
                        "output volume of (get volume settings)"])
            m = re.search(r"\d+", out)
            if not m:
                raise RuntimeError(f"could not parse osascript volume: {out!r}")
            return _clamp(int(m.group()))
        if self.backend == "wpctl":
            out = _run(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"])
            m = re.search(r"(\d+(?:\.\d+)?)%", out)
            if not m:
                raise RuntimeError(f"could not parse wpctl volume: {out!r}")
            return _clamp(round(float(m.group(1))))
        if self.backend == "amixer":
            out = _run(["amixer", "sget", "Master"])
            m = re.search(r"\[(\d+)%\]", out)
            if not m:
                raise RuntimeError(f"could not parse amixer volume: {out!r}")
            return _clamp(int(m.group(1)))
        raise RuntimeError(f"no usable volume backend ({self.backend})")

    def _set_level(self, level: int) -> None:
        level = _clamp(level)
        if self._set_override is not None:
            self._set_override(level)
            return
        if self.backend == "osascript":
            _run(["osascript", "-e", f"set volume output volume {level}"])
        elif self.backend == "wpctl":
            _run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{level}%"])
        elif self.backend == "amixer":
            _run(["amixer", "-q", "sset", "Master", f"{level}%"])
        else:
            raise RuntimeError(f"no usable volume backend ({self.backend})")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def begin(self) -> int | None:
        """Snapshot the current output volume at demo start.

        Returns the captured level, or ``None`` if it could not be read
        (in which case there is nothing to restore later).
        """
        try:
            self._captured = self._get_level()
            self._restored = False
            LOGGER.info("[volume] captured pre-demo level=%s%% (backend=%s)",
                        self._captured, self.backend)
            return self._captured
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[volume] could not capture level: %s", exc)
            self._captured = None
            return None

    def get(self) -> int:
        return self._get_level()

    def set(self, level: int) -> int:
        """Set an absolute output level; returns the clamped level applied."""
        level = _clamp(level)
        self._set_level(level)
        LOGGER.info("[volume] set level=%s%%", level)
        return level

    def step(self, delta: int) -> int:
        """Move the volume by ``delta`` percent; returns the new level."""
        new_level = _clamp(self.get() + int(delta))
        self.set(new_level)
        return new_level

    def mute(self) -> None:
        """Silence the output without losing the remembered level.

        The captured pre-demo level is preserved, so :meth:`end` (or an
        explicit :meth:`restore`) brings it back.
        """
        self._set_level(0)
        LOGGER.info("[volume] muted")

    def unmute(self) -> int:
        """Return the output to the last known non-zero level.

        Falls back to the pre-demo snapshot if no other level is remembered.
        """
        target = self._captured if self._captured is not None else MAX_LEVEL
        self.set(target)
        return target

    def restore(self) -> bool:
        """Restore the pre-demo volume. Idempotent; safe to call repeatedly."""
        if self._restored:
            return True
        if self._captured is None:
            LOGGER.info("[volume] nothing to restore (no captured level)")
            return False
        try:
            self._set_level(self._captured)
            self._restored = True
            LOGGER.info("[volume] restored pre-demo level=%s%%", self._captured)
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[volume] restore failed: %s", exc)
            return False

    def end(self) -> bool:
        """Demo teardown hook: restore and report success."""
        return self.restore()
