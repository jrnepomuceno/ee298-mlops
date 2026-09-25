"""Optional offline Piper TTS and ALSA playback."""
from __future__ import annotations

import subprocess
import os
from typing import Any


class WavPlayer:
    """Play pre-generated replies without loading or invoking Piper."""

    def __init__(self, player: str = "pw-play",
                 audio_device: str | None = None) -> None:
        self.player = player
        self.audio_device = audio_device
        self._async_processes: list[subprocess.Popen[Any]] = []

    def _command(self, path: str) -> list[str]:
        player_command = [self.player]
        if os.path.basename(self.player) in {"pw-play", "aplay"}:
            player_command.append("-q")
        if self.audio_device:
            if os.path.basename(self.player) == "pw-play":
                player_command.append(f"--target={self.audio_device}")
            else:
                player_command.extend(["-D", self.audio_device])
        player_command.append(path)
        return player_command

    def play(self, path: str) -> dict[str, Any]:
        player_command = self._command(path)
        try:
            subprocess.run(player_command, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, check=True, timeout=60)
        except (FileNotFoundError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"WAV playback failed: {exc}") from exc
        return {"status": "played", "wav": path, "player": player_command}

    def play_async(self, path: str) -> dict[str, Any]:
        """Start playback and return without waiting for the WAV to finish."""
        self._reap_async_processes()
        player_command = self._command(path)
        try:
            process = subprocess.Popen(
                player_command, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except OSError as exc:
            raise RuntimeError(f"WAV playback failed: {exc}") from exc
        self._async_processes.append(process)
        return {"status": "started", "wav": path, "player": player_command}

    def close(self) -> None:
        """Reap outstanding asynchronous playback processes."""
        for process in self._async_processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=2)
        self._async_processes.clear()

    def _reap_async_processes(self) -> None:
        self._async_processes = [
            process for process in self._async_processes
            if process.poll() is None
        ]


class PiperTts:
    """Generate a WAV with the locally installed Piper voice."""

    def __init__(self, executable: str, model: str) -> None:
        self.executable = executable
        self.model = model

    def synthesize(self, text: str, output_path: str) -> None:
        command = [
            self.executable,
            "--model", self.model,
            "--output_file", output_path,
        ]
        try:
            subprocess.run(
                command,
                input=text + "\n",
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                timeout=30,
            )
        except (OSError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Piper synthesis failed: {exc}") from exc
