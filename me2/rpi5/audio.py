"""Microphone capture and small energy-based VAD for the Pi runtime."""
from __future__ import annotations

from collections import deque
from contextlib import closing
from dataclasses import dataclass
from queue import Queue
import shutil
import subprocess
import time
from datetime import datetime
from typing import Any, Iterator

import numpy as np


def log(message: str) -> None:
    """Write a timestamped microphone log line."""
    timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
    print(f"{timestamp}\t{message}", flush=True)


@dataclass(frozen=True)
class VADConfig:
    sample_rate: int = 16_000
    capture_sample_rate: int = 44_100
    capture_device: str | None = None
    frame_ms: int = 30
    start_rms: float = 0.025
    stop_rms: float = 0.015
    start_frames: int = 2
    silence_frames: int = 12
    preroll_frames: int = 3
    max_utterance_s: float = 5.0


class EnergyVAD:
    """Turn float32 microphone frames into complete utterances."""

    def __init__(self, config: VADConfig | None = None) -> None:
        self.config = config or VADConfig()
        self.frame_samples = (self.config.sample_rate
                              * self.config.frame_ms // 1000)
        self.capture_frame_samples = (self.config.capture_sample_rate
                                       * self.config.frame_ms // 1000)
        self._pre = deque(maxlen=self.config.preroll_frames)
        self._frames: list[np.ndarray] = []
        self._speech_frames = 0
        self._quiet_frames = 0

    def accept(self, frame: np.ndarray) -> np.ndarray | None:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if frame.size == 0:
            return None
        rms = float(np.sqrt(np.mean(frame * frame)))
        if not self._frames:
            self._pre.append(frame.copy())

        if not self._frames:
            if rms >= self.config.start_rms:
                self._speech_frames += 1
            else:
                self._speech_frames = 0
            if self._speech_frames >= self.config.start_frames:
                self._frames = list(self._pre)
                self._quiet_frames = 0
            return None

        self._frames.append(frame.copy())
        if rms < self.config.stop_rms:
            self._quiet_frames += 1
        else:
            self._quiet_frames = 0

        too_long = len(self._frames) * self.config.frame_ms / 1000 >= self.config.max_utterance_s
        if self._quiet_frames >= self.config.silence_frames or too_long:
            utterance = np.concatenate(self._frames).astype(np.float32)
            self.reset()
            return utterance
        return None

    def flush(self) -> np.ndarray | None:
        if not self._frames:
            return None
        utterance = np.concatenate(self._frames).astype(np.float32)
        self.reset()
        return utterance

    def reset(self) -> None:
        self._pre.clear()
        self._frames = []
        self._speech_frames = 0
        self._quiet_frames = 0

    @property
    def active(self) -> bool:
        return bool(self._frames)


def microphone_utterances(config: VADConfig | None = None,
                          wakeword: Any | None = None,
                          on_wake: Any | None = None,
                          on_speech: Any | None = None,
                          on_timeout: Any | None = None,
                          cooldown_s: float = 1.0,
                          command_timeout_s: float = 7.0,
                          wake_only: bool = False) -> Iterator[np.ndarray]:
    """Yield utterances from the default microphone until interrupted.

    ``sounddevice`` is imported lazily so replay/self-test mode does not need a
    recording device or the sounddevice package.
    """
    vad = EnergyVAD(config)
    wakeword_active = wakeword is None
    speech_announced = False
    command_deadline: float | None = None
    settings = vad.config
    frames: Queue[np.ndarray] = Queue()

    def resample_frame(frame: np.ndarray, source_rate: int) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if source_rate == settings.sample_rate:
            return frame
        target_size = round(frame.size * settings.sample_rate / source_rate)
        source_x = np.linspace(0.0, 1.0, frame.size, endpoint=False)
        target_x = np.linspace(0.0, 1.0, target_size, endpoint=False)
        return np.interp(target_x, source_x, frame).astype(np.float32)

    pipewire = shutil.which("pw-record") is not None
    if pipewire:
        capture_source = _pipewire_capture(settings)
        capture_context = closing(capture_source)
        get_frame = lambda: resample_frame(next(capture_source), 48000)
        discard_frame = get_frame
    else:
        capture_context = _sounddevice_capture(
            settings, vad.capture_frame_samples, frames
        )

        def get_frame() -> np.ndarray:
            return frames.get()

        def discard_frame() -> np.ndarray:
            return frames.get()

    with capture_context:
        while True:
            frame = get_frame()
            if not wakeword_active:
                if not wakeword.accepts(frame):
                    continue
                wakeword_active = True
                if on_wake is not None:
                    on_wake()
                vad.reset()
                if cooldown_s > 0:
                    deadline = time.monotonic() + cooldown_s
                    while time.monotonic() < deadline:
                        discard_frame()
                speech_announced = True
                if wake_only:
                    wakeword_active = False
                    speech_announced = False
                    continue
                command_deadline = time.monotonic() + command_timeout_s
                continue
            if (command_deadline is not None and not vad.active
                    and time.monotonic() >= command_deadline):
                vad.reset()
                command_deadline = None
                speech_announced = False
                if on_timeout is not None:
                    on_timeout()
                if wakeword is not None and hasattr(wakeword, "reset"):
                    wakeword.reset()
                if cooldown_s > 0:
                    deadline = time.monotonic() + cooldown_s
                    while time.monotonic() < deadline:
                        discard_frame()
                wakeword_active = wakeword is None
                continue
            utterance = vad.accept(frame)
            if vad.active and not speech_announced:
                speech_announced = True
                if on_speech is not None:
                    on_speech()
            if utterance is not None:
                command_deadline = None
                wakeword_active = wakeword is None
                speech_announced = False
                yield utterance
                # TTS can be audible to the microphone. Discard frames queued
                # during playback and give the room time to become quiet.
                if cooldown_s > 0:
                    deadline = time.monotonic() + cooldown_s
                    while time.monotonic() < deadline:
                        discard_frame()
                vad.reset()
                if wakeword is not None and hasattr(wakeword, "reset"):
                    wakeword.reset()
                wakeword_active = wakeword is None


def _pipewire_capture(settings: VADConfig) -> Iterator[np.ndarray]:
    """Yield mono PCM frames from the PipeWire default capture source."""
    command = [
        "pw-record", "--raw", "--rate=48000", "--channels=1",
        "--format=s16",
    ]
    if settings.capture_device:
        command.extend(["--target", settings.capture_device])
    command.append("-")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    assert process.stdout is not None
    frame_bytes = round(48000 * settings.frame_ms / 1000) * 2
    try:
        while True:
            data = process.stdout.read(frame_bytes)
            if len(data) != frame_bytes:
                return
            yield np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()


def _sounddevice_capture(settings: VADConfig, frame_samples: int,
                         frames: Queue[np.ndarray]) -> Any:
    """Return the legacy PortAudio capture context when PipeWire is absent."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "microphone mode requires pw-record or sounddevice; "
            "install requirements-runtime.txt"
        ) from exc

    def on_audio(indata, _frames, _time, status) -> None:
        if status:
            log(f"[audio] {status}")
        frame = np.asarray(indata, dtype=np.float32).reshape(-1)
        frames.put(frame)

    return sd.InputStream(
        device=settings.capture_device,
        samplerate=settings.capture_sample_rate,
        blocksize=frame_samples,
        channels=1,
        dtype="float32",
        callback=on_audio,
    )
