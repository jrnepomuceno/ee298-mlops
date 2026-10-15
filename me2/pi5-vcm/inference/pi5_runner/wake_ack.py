"""Alexa wake-word acknowledgement runtime for the Raspberry Pi 5."""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO, Protocol

import numpy as np

SAMPLE_RATE = 16_000
FRAME_MILLISECONDS = 80
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MILLISECONDS // 1_000
FRAME_BYTES = FRAME_SAMPLES * 2


class WakeWordModel(Protocol):
    def predict(self, audio: np.ndarray) -> dict[str, float]: ...

    def reset(self) -> None: ...


class AudioCapture(Protocol):
    def read_frame(self) -> bytes | None: ...

    def close(self) -> None: ...


class Indicator(Protocol):
    def turn_on(self) -> None: ...

    def turn_off(self) -> None: ...

    def close(self) -> None: ...


class BeepPlayer(Protocol):
    def play(self) -> None: ...


def read_exact(stream: BinaryIO, size: int) -> bytes | None:
    """Read one complete frame, returning None only at a clean end of stream."""
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if not chunks:
                return None
            raise RuntimeError(
                f"ALSA capture ended in the middle of a frame "
                f"({size - remaining} of {size} bytes)"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class AlsaCapture:
    """Read mono 16-bit PCM frames from an ALSA arecord subprocess."""

    def __init__(
        self,
        *,
        device: str,
        executable: str = "arecord",
        frame_bytes: int = FRAME_BYTES,
    ) -> None:
        self.device = device
        self.executable = executable
        self.frame_bytes = frame_bytes
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_file: BinaryIO | None = None

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("ALSA capture is already running")
        executable = shutil.which(self.executable)
        if executable is None:
            raise RuntimeError(
                f"{self.executable!r} was not found; install ALSA utilities"
            )
        self._stderr_file = tempfile.TemporaryFile()
        try:
            self._process = subprocess.Popen(
                [
                    executable,
                    "--quiet",
                    "--device",
                    self.device,
                    "--file-type",
                    "raw",
                    "--format",
                    "S16_LE",
                    "--rate",
                    str(SAMPLE_RATE),
                    "--channels",
                    "1",
                ],
                stdout=subprocess.PIPE,
                stderr=self._stderr_file,
            )
        except OSError:
            self._stderr_file.close()
            self._stderr_file = None
            raise

    def read_frame(self) -> bytes | None:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("ALSA capture has not been started")
        frame = read_exact(self._process.stdout, self.frame_bytes)
        if frame is None:
            try:
                return_code = self._process.wait(timeout=1)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "ALSA capture closed its audio stream but did not exit"
                ) from exc
            details = self._read_stderr()
            raise RuntimeError(
                f"ALSA capture exited with status {return_code}"
                + (f": {details}" if details else "")
            )
        return frame

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdout is not None:
                process.stdout.close()
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None

    def _read_stderr(self) -> str:
        if self._stderr_file is None:
            return ""
        self._stderr_file.seek(0)
        return self._stderr_file.read().decode("utf-8", errors="replace").strip()


class OpenWakeWordDetector:
    def __init__(
        self,
        model_path: Path,
        *,
        model: WakeWordModel | None = None,
    ) -> None:
        if model is None:
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Alexa wake-word model does not exist: {model_path}"
                )
            extension = model_path.suffix.lower()
            if extension not in {".onnx", ".tflite"}:
                raise ValueError(
                    "Alexa wake-word model must be an ONNX or TFLite file"
                )
            try:
                from openwakeword.model import Model
            except ImportError as exc:
                raise RuntimeError(
                    "openWakeWord is required; see inference/README.md for "
                    "the Pi-specific installation requirements"
                ) from exc
            framework = "onnx" if extension == ".onnx" else "tflite"
            feature_extension = ".onnx" if framework == "onnx" else ".tflite"
            model = Model(
                wakeword_models=[str(model_path)],
                inference_framework=framework,
                melspec_model_path=str(
                    model_path.with_name(f"melspectrogram{feature_extension}")
                ),
                embedding_model_path=str(
                    model_path.with_name(f"embedding_model{feature_extension}")
                ),
            )
        self.model = model

    def reset(self) -> None:
        self.model.reset()

    def score_frame(self, frame: bytes) -> float:
        if len(frame) != FRAME_BYTES:
            raise ValueError(
                f"wake-word input must be exactly {FRAME_BYTES} bytes "
                f"({FRAME_MILLISECONDS} ms of mono PCM)"
            )
        audio = np.frombuffer(frame, dtype="<i2")
        scores = self.model.predict(audio)
        if not scores:
            raise RuntimeError("openWakeWord returned no wake-word scores")
        score = max(float(value) for value in scores.values())
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError(f"openWakeWord returned an invalid score: {score}")
        return score


class GpioIndicator:
    def __init__(self, pin: int) -> None:
        try:
            from gpiozero import LED
        except ImportError as exc:
            raise RuntimeError(
                "gpiozero is required for the indicator; install it on the Pi"
            ) from exc
        self._led = LED(pin)

    def turn_on(self) -> None:
        self._led.on()

    def turn_off(self) -> None:
        self._led.off()

    def close(self) -> None:
        self._led.close()


def make_beep_pcm(
    *,
    sample_rate: int = 48_000,
    frequency_hz: float = 880.0,
    duration_seconds: float = 0.18,
    amplitude: float = 0.18,
) -> bytes:
    if sample_rate <= 0 or frequency_hz <= 0 or duration_seconds <= 0:
        raise ValueError("beep rate, frequency, and duration must be positive")
    if not 0.0 < amplitude <= 1.0:
        raise ValueError("beep amplitude must be in (0, 1]")

    sample_count = round(sample_rate * duration_seconds)
    fade_samples = min(round(sample_rate * 0.005), sample_count // 2)
    pcm = bytearray()
    for index in range(sample_count):
        envelope = 1.0
        if fade_samples:
            if index < fade_samples:
                envelope = index / fade_samples
            elif index >= sample_count - fade_samples:
                envelope = (sample_count - 1 - index) / fade_samples
        value = amplitude * envelope * math.sin(
            2.0 * math.pi * frequency_hz * index / sample_rate
        )
        pcm.extend(struct.pack("<h", round(value * 32767)))
    return bytes(pcm)


class AudioBeepPlayer:
    def __init__(
        self,
        *,
        device: str | None,
        alsa_executable: str = "aplay",
        pipewire_executable: str = "pw-play",
    ) -> None:
        self.device = device
        self.alsa_executable = alsa_executable
        self.pipewire_executable = pipewire_executable

    def play(self) -> None:
        pipewire_target: str | None = None
        if self.device is not None and self.device.startswith("pipewire:"):
            pipewire_target = self.device.removeprefix("pipewire:") or "auto"
            executable_name = self.pipewire_executable
        else:
            executable_name = self.alsa_executable
        executable = shutil.which(executable_name)
        if executable is None:
            raise RuntimeError(
                f"{executable_name!r} was not found; install the required audio tools"
            )
        if pipewire_target is not None:
            command = [
                executable,
                "--raw",
                "--format",
                "s16",
                "--rate",
                "48000",
                "--channels",
                "1",
                "--target",
                pipewire_target,
                "-",
            ]
        else:
            command = [
                executable,
                "--quiet",
                "--file-type",
                "raw",
                "--format",
                "S16_LE",
                "--rate",
                "48000",
                "--channels",
                "1",
            ]
            if self.device:
                command.extend(["--device", self.device])
        try:
            result = subprocess.run(
                command,
                input=make_beep_pcm(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
                timeout=5,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("beep playback timed out") from exc
        if result.returncode != 0:
            details = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"beep playback failed with status {result.returncode}"
                + (f": {details}" if details else "")
            )


def listen_for_wake(
    detector: OpenWakeWordDetector,
    capture: AudioCapture,
    *,
    threshold: float,
) -> float:
    while True:
        frame = capture.read_frame()
        if frame is None:
            raise RuntimeError("audio capture ended before a wake word was detected")
        score = detector.score_frame(frame)
        if score >= threshold:
            return score


def acknowledge_wake(
    capture: AudioCapture,
    indicator: Indicator | None,
    beep_player: BeepPlayer,
) -> None:
    capture.close()
    if indicator is not None:
        indicator.turn_on()
    beep_player.play()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listen for Alexa and acknowledge detection with an LED and beep."
    )
    parser.add_argument("--wakeword-model", type=Path, required=True)
    parser.add_argument(
        "--input-device",
        default="default:CARD=DuoCast",
        help="ALSA capture device (the Pi 5 currently has a HyperX DuoCast)",
    )
    parser.add_argument(
        "--output-device",
        help=(
            "ALSA playback device or pipewire:<target> (for example, "
            "pipewire:auto); defaults to ALSA's system playback device"
        ),
    )
    led = parser.add_mutually_exclusive_group(required=True)
    led.add_argument(
        "--led-gpio",
        type=int,
        help="BCM GPIO pin connected to an external LED through a resistor",
    )
    led.add_argument(
        "--no-led",
        action="store_true",
        help="run without a GPIO indicator (for audio-only tests)",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--indicator-hold-seconds",
        type=float,
        default=1.0,
        help="keep the indicator on briefly after each beep",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="exit after one wake acknowledgement instead of listening continuously",
    )
    args = parser.parse_args(argv)
    if not 0.0 < args.threshold <= 1.0:
        parser.error("--threshold must be in (0, 1]")
    if (
        not math.isfinite(args.indicator_hold_seconds)
        or args.indicator_hold_seconds < 0.0
    ):
        parser.error("--indicator-hold-seconds must be finite and not negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    logger = logging.getLogger("pi5_vcm.inference")
    capture = AlsaCapture(device=args.input_device)
    detector = OpenWakeWordDetector(args.wakeword_model)
    indicator = None if args.no_led else GpioIndicator(args.led_gpio)
    beep_player = AudioBeepPlayer(device=args.output_device)
    try:
        logger.info(
            "wake listener started: wake_word=alexa input=%s output=%s "
            "threshold=%.3f continuous=%s",
            args.input_device,
            args.output_device or "default",
            args.threshold,
            not args.once,
        )
        while True:
            capture.start()
            score = listen_for_wake(detector, capture, threshold=args.threshold)
            acknowledge_wake(capture, indicator, beep_player)
            logger.info(
                "wake acknowledged: wake_word=alexa score=%.4f",
                score,
            )
            if indicator is not None:
                time.sleep(args.indicator_hold_seconds)
                indicator.turn_off()
            if args.once:
                return 0
            detector.reset()
            logger.info("wake listener rearmed")
    except KeyboardInterrupt:
        logger.info("wake listener stopped")
        return 0
    finally:
        capture.close()
        if indicator is not None:
            indicator.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logging.getLogger("pi5_vcm.inference").exception(
            "wake acknowledgement runtime failed: %s",
            exc,
        )
        raise SystemExit(1) from exc
