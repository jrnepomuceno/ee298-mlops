"""Optional HyperX DuoCast RGB status controller."""
from __future__ import annotations

import subprocess
from typing import Any


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