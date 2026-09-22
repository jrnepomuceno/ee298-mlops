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

    def play(self, path: str) -> dict[str, Any]:
        player_command = [self.player]
        if os.path.basename(self.player) in {"pw-play", "aplay"}:
            player_command.append("-q")
        if self.audio_device:
            if os.path.basename(self.player) == "pw-play":
                player_command.append(f"--target={self.audio_device}")
            else:
                player_command.extend(["-D", self.audio_device])
        player_command.append(path)
        try:
            subprocess.run(player_command, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, check=True, timeout=60)
        except (FileNotFoundError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"WAV playback failed: {exc}") from exc
        return {"status": "played", "wav": path, "player": player_command}
