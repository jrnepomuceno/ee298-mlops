"""Optional offline Piper TTS and ALSA playback."""
from __future__ import annotations

import subprocess
import os
import tempfile
import wave
from typing import Any

PRE_ROLL_MS = 750


def _with_silence_preroll(path: str, preroll_ms: int = PRE_ROLL_MS) -> str:
    """Write a PCM WAV copy with digital silence before the original samples."""
    try:
        source = wave.open(path, "rb")
    except (OSError, wave.Error) as exc:
        raise RuntimeError(f"cannot read playback WAV {path}: {exc}") from exc

    temp_path = ""
    try:
        with source:
            params = source.getparams()
            if params.comptype != "NONE":
                raise RuntimeError(f"unsupported compressed playback WAV: {path}")
            silence_frames = round(params.framerate * preroll_ms / 1000.0)
            silence_sample = (b"\x80" if params.sampwidth == 1
                              else bytes(params.sampwidth))
            silence = silence_sample * silence_frames * params.nchannels
            with tempfile.NamedTemporaryFile(prefix="pi5-vcm-preroll-",
                                             suffix=".wav", delete=False) as temp:
                temp_path = temp.name
            with wave.open(temp_path, "wb") as padded:
                padded.setnchannels(params.nchannels)
                padded.setsampwidth(params.sampwidth)
                padded.setframerate(params.framerate)
                padded.writeframesraw(silence)
                while chunk := source.readframes(65536):
                    padded.writeframesraw(chunk)
        return temp_path
    except Exception:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise


class WavPlayer:
    """Play pre-generated replies without loading or invoking Piper."""

    def __init__(self, player: str = "pw-play",
                 audio_device: str | None = None) -> None:
        self.player = player
        self.audio_device = audio_device
        self._async_processes: list[tuple[subprocess.Popen[Any], str]] = []

    def _command(self, path: str) -> list[str]:
        player_command = [self.player]
        if os.path.basename(self.player) == "aplay":
            player_command.append("-q")
        if self.audio_device:
            if os.path.basename(self.player) == "pw-play":
                player_command.append(f"--target={self.audio_device}")
            else:
                player_command.extend(["-D", self.audio_device])
        player_command.append(path)
        return player_command

    def play(self, path: str, *, preroll_ms: int = PRE_ROLL_MS) -> dict[str, Any]:
        playback_path = _with_silence_preroll(path, preroll_ms)
        player_command = self._command(playback_path)
        try:
            subprocess.run(player_command, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, check=True, timeout=60)
        except (FileNotFoundError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"WAV playback failed: {exc}") from exc
        finally:
            os.unlink(playback_path)
        return {"status": "played", "wav": path, "player": player_command,
                "preroll_ms": preroll_ms}

    def play_async(self, path: str) -> dict[str, Any]:
        """Start playback and return without waiting for the WAV to finish."""
        self._reap_async_processes()
        playback_path = _with_silence_preroll(path)
        player_command = self._command(playback_path)
        try:
            process = subprocess.Popen(
                player_command, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except OSError as exc:
            os.unlink(playback_path)
            raise RuntimeError(f"WAV playback failed: {exc}") from exc
        self._async_processes.append((process, playback_path))
        return {"status": "started", "wav": path, "player": player_command,
                "preroll_ms": PRE_ROLL_MS}

    def close(self) -> None:
        """Reap outstanding asynchronous playback processes."""
        for process, playback_path in self._async_processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=2)
            try:
                os.unlink(playback_path)
            except FileNotFoundError:
                pass
        self._async_processes.clear()

    def _reap_async_processes(self) -> None:
        pending = []
        for process, playback_path in self._async_processes:
            if process.poll() is None:
                pending.append((process, playback_path))
            else:
                try:
                    os.unlink(playback_path)
                except FileNotFoundError:
                    pass
        self._async_processes = pending


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
