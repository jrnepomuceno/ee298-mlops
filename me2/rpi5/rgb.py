"""Optional HyperX DuoCast RGB status controller."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import Any, ClassVar


class RgbController:
    """Map assistant states to QuadcastRGB commands.

    The default is dry-run. Set ``executable`` to ``quadcastrgb`` to control
    the physical DuoCast. The cycle process is stopped before a new state is
    applied, preventing multiple USB controllers from competing.
    """

    def __init__(self, executable: str | None = None) -> None:
        self.executable = executable
        self._cycle: subprocess.Popen[Any] | None = None

    @property
    def enabled(self) -> bool:
        return self.executable is not None

    def standby(self) -> dict[str, Any]:
        return self.idle()

    def idle(self) -> dict[str, Any]:
        return self.solid("000000", "idle")

    def wake(self) -> dict[str, Any]:
        return self.animate("00A0FF", "listening", speed=10)

    def warming(self, color: str = "FFA500", speed: int = 8) -> dict[str, Any]:
        """Fast breathing (pulse) while the checkpoint loads and warms."""
        return self.animate(color, "warming", speed=speed, mode="pulse")

    def processing(self) -> dict[str, Any]:
        return self.solid("FF8C00", "processing")

    def acting(self) -> dict[str, Any]:
        return self.solid("8A2BE2", "acting")

    def speaking(self) -> dict[str, Any]:
        return self.animate("00FF00", "tts", speed=3)

    def off(self) -> dict[str, Any]:
        return self.idle()

    def solid(self, color: str, state: str) -> dict[str, Any]:
        self._stop_cycle()
        self._stop_quadcast_process()
        return self._run("solid", color, state=state)

    def cycle(self, state: str) -> dict[str, Any]:
        self._stop_cycle()
        self._stop_quadcast_process()
        if not self.enabled:
            return {"status": "dry_run", "state": state, "command": None}
        command = [self.executable, "cycle"]
        self._cycle = subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return {"status": "applied", "state": state, "command": command}

    def animate(self, color: str, state: str, speed: int,
                mode: str = "wave") -> dict[str, Any]:
        self._stop_cycle()
        self._stop_quadcast_process()
        if not self.enabled:
            return {"status": "dry_run", "state": state, "command": None}
        command = [self.executable, "-s", str(speed), mode, color]
        self._cycle = subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return {"status": "applied", "state": state, "command": command}

    def close(self) -> None:
        self.idle()

    def _run(self, mode: str, value: str, state: str) -> dict[str, Any]:
        command = [self.executable, mode, value] if self.enabled else None
        if command is None:
            return {"status": "dry_run", "state": state, "command": None}
        try:
            subprocess.run(command, check=True, timeout=5,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError) as exc:
            return {"status": "error", "state": state, "command": command,
                    "error": str(exc)}
        return {"status": "applied", "state": state, "command": command}

    def _stop_cycle(self) -> None:
        if self._cycle is not None and self._cycle.poll() is None:
            self._cycle.terminate()
            try:
                self._cycle.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._cycle.kill()
        self._cycle = None

    @staticmethod
    def _stop_quadcast_process() -> None:
        subprocess.run(
            ["killall", "quadcastrgb"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


# --------------------------------------------------------------------------- #
# Pluggable light drivers
#
# The :class:`RgbController` above is the *demo-lifecycle* indicator: it maps
# assistant states (listening / processing / speaking) onto the DuoCast ring
# and is driven directly by ``run.py``. The voice-command intents
# (turn_on_lights / turn_off_lights / dim_lights) instead go through the
# executor layer, which talks to a *driver* chosen at startup. That split is
# deliberate: the ring can wear two hats at once, and swapping the light
# device behind a voice command never touches the lifecycle indicator.
#
# A driver is anything implementing :class:`LightDriver`. Add a new device by
# subclassing it and registering a name in :data:`LIGHT_DRIVERS`; selecting it
# is then a single CLI flag (``--light-driver NAME``), no code change.
# --------------------------------------------------------------------------- #


class LightDriver(ABC):
    """Common surface for a controllable light device.

    Drivers are stateful (they remember the last brightness so a bare
    ``on`` can restore it) and are expected to be cheap to construct. The
    executor wraps them in a try/except, so a driver may raise on a real
    hardware fault; it should not raise for "device absent" -- that is
    reported via :meth:`available`.
    """

    #: human-readable name used in CLI selection and diagnostics
    name: ClassVar[str] = "base"

    @abstractmethod
    def on(self) -> None:
        """Turn the light on (restore the last brightness, or a default)."""

    @abstractmethod
    def off(self) -> None:
        """Turn the light off."""

    @abstractmethod
    def set_brightness(self, percent: int) -> int:
        """Set brightness to ``percent`` (1-100) and return the applied value."""

    def available(self) -> bool:
        """Whether the underlying device/tool is present on this host."""
        return True

    def close(self) -> None:
        """Release resources; safe to call repeatedly."""


class HyperxDuoCastDriver(LightDriver):
    """Drive the HyperX DuoCast microphone ring light via ``quadcastrgb``.

    ``quadcastrgb`` is the community CLI for the DuoCast/QuadCast S RGB ring.
    Its ``solid <RRGGBB>`` form writes a constant colour and, on the DuoCast,
    brightness is expressed by scaling the colour toward black (the ring has
    no separate brightness channel). The executable is resolved from
    ``$QUADCASTRGB`` (if set) or ``PATH``.

    When the tool is absent the driver degrades to a no-op that reports
    ``available() == False`` rather than raising, so the assistant can still
    answer honestly ("the light isn't reachable") instead of erroring.
    """

    name = "hyperx-duocast"

    #: brightness used when the user says "turn the lights on" with no level
    DEFAULT_BRIGHTNESS = 100
    #: warm white used for a lit ring; black is "off"
    LIT_COLOR = "FFF4E6"

    def __init__(self, executable: str | None = None) -> None:
        self.executable = executable or os.environ.get("QUADCASTRGB") or "quadcastrgb"
        self._brightness = self.DEFAULT_BRIGHTNESS

    def available(self) -> bool:
        if self.executable.startswith("/"):
            return os.path.exists(self.executable)
        return shutil.which(self.executable) is not None

    def on(self) -> None:
        self.set_brightness(self._brightness)

    def off(self) -> None:
        self._solid("000000")

    def set_brightness(self, percent: int) -> int:
        percent = max(1, min(100, int(percent)))
        self._brightness = percent
        self._solid(self._scale_color(self.LIT_COLOR, percent))
        return percent

    def close(self) -> None:
        self.off()

    # -- internals --------------------------------------------------------- #

    def _scale_color(self, hexcolor: str, percent: int) -> str:
        r = int(hexcolor[0:2], 16)
        g = int(hexcolor[2:4], 16)
        b = int(hexcolor[4:6], 16)
        f = percent / 100.0
        return "%02X%02X%02X" % (round(r * f), round(g * f), round(b * f))

    def _solid(self, hexcolor: str) -> None:
        if not self.available():
            return  # degrade quietly; availability is surfaced by the executor
        subprocess.run(
            [self.executable, "solid", hexcolor],
            check=True, timeout=5,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


#: Registry of selectable drivers. ``make_light_driver(name)`` looks names up
#: here, so adding a device is one line plus one class.
LIGHT_DRIVERS: dict[str, type[LightDriver]] = {
    HyperxDuoCastDriver.name: HyperxDuoCastDriver,
}


def make_light_driver(name: str | None, executable: str | None = None) -> LightDriver | None:
    """Instantiate a driver by registered name.

    Returns ``None`` for a falsy name (lights stay dry-run) or for an unknown
    name (logged to stderr so a typo is visible, and lights fall back to
    dry-run rather than crashing the mic loop).
    """
    if not name:
        return None
    cls = LIGHT_DRIVERS.get(name)
    if cls is None:
        print(f"[lights] unknown driver '{name}'; "
              f"known: {', '.join(sorted(LIGHT_DRIVERS))}; staying dry-run",
              file=sys.stderr)
        return None
    return cls(executable)