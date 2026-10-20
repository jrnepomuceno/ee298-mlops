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
from threading import Lock
import time
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO, Protocol

import numpy as np

if __package__:
    from .intent_execution import INTENT_NAMES as EXECUTION_INTENT_NAMES
    from .intent_execution import (
        CREATE_REMINDER_ACTIONS,
        CREATE_REMINDER_SLOT_INDEX,
        ALARM_ACTIONS,
        ALARM_SLOT_INDEX,
        TIMER_ACTIONS,
        TIMER_EXPIRED_MESSAGE,
        TIMER_SLOT_INDEX,
    )
    from .intent_execution import IntentDispatcher
    from .intent_inference import OnnxIntentPredictor
else:
    from intent_execution import INTENT_NAMES as EXECUTION_INTENT_NAMES
    from intent_execution import (
        CREATE_REMINDER_ACTIONS,
        CREATE_REMINDER_SLOT_INDEX,
        ALARM_ACTIONS,
        ALARM_SLOT_INDEX,
        TIMER_ACTIONS,
        TIMER_EXPIRED_MESSAGE,
        TIMER_SLOT_INDEX,
    )
    from intent_execution import IntentDispatcher
    from intent_inference import OnnxIntentPredictor

SAMPLE_RATE = 16_000
FRAME_MILLISECONDS = 80
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MILLISECONDS // 1_000
FRAME_BYTES = FRAME_SAMPLES * 2
COMMAND_START_TIMEOUT_SECONDS = 3.0
COMMAND_MAX_DURATION_SECONDS = 8.0
COMMAND_END_SILENCE_SECONDS = 0.8
COMMAND_MIN_VOICED_SECONDS = 0.24
COMMAND_VOICE_THRESHOLD_DBFS = -42.0
WAKE_MODEL_WARMUP_FRAMES = 3
PIPER_WARMUP_TEXT = "Ready."


class WakeWordModel(Protocol):
    def predict(self, audio: np.ndarray) -> dict[str, float]: ...

    def reset(self) -> None: ...


class AudioCapture(Protocol):
    def read_frame(self) -> bytes | None: ...

    def close(self) -> None: ...


class Indicator(Protocol):
    def start_warmup(self) -> None: ...

    def finish_warmup(self) -> None: ...

    def turn_on(self) -> None: ...

    def speaking(self) -> None: ...

    def turn_off(self) -> None: ...

    def close(self) -> None: ...


class BeepPlayer(Protocol):
    def play(self) -> None: ...


class TtsVoiceConfig(Protocol):
    sample_rate: int


class TtsVoice(Protocol):
    config: TtsVoiceConfig

    def synthesize_wav(self, text: str, wav_file: wave.Wave_write) -> object: ...


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

    def warmup(self) -> None:
        silence_frame = bytes(FRAME_BYTES)
        try:
            for _ in range(WAKE_MODEL_WARMUP_FRAMES):
                self.score_frame(silence_frame)
        finally:
            self.reset()

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

    def start_warmup(self) -> None:
        self.turn_on()

    def finish_warmup(self) -> None:
        self.turn_off()

    def turn_on(self) -> None:
        self._led.on()

    def speaking(self) -> None:
        self._led.on()

    def turn_off(self) -> None:
        self._led.off()

    def close(self) -> None:
        self._led.close()


class HyperxRgbIndicator:
    def __init__(self, executable: Path) -> None:
        if not executable.is_file() or not executable.stat().st_mode & 0o111:
            raise FileNotFoundError(
                "foreground QuadcastRGB executable is missing or not executable: "
                f"{executable}"
            )
        self.executable = executable
        self._process: subprocess.Popen[bytes] | None = None
        self._mode: tuple[str, ...] | None = None

    def start_warmup(self) -> None:
        self._set_mode("-a", "cycle")

    def finish_warmup(self) -> None:
        self.turn_off()

    def turn_on(self) -> None:
        self._set_mode("-a", "solid", "0000ff")

    def speaking(self) -> None:
        self._set_mode("-a", "solid", "00ff00")

    def turn_off(self) -> None:
        self._set_mode("-a", "solid", "0")

    def close(self) -> None:
        try:
            self.turn_off()
        finally:
            self._stop()

    def _set_mode(self, *arguments: str) -> None:
        if (
            self._mode == arguments
            and self._process is not None
            and self._process.poll() is None
        ):
            return
        self._stop()
        try:
            process = subprocess.Popen(
                [str(self.executable), *arguments],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise RuntimeError(f"could not start QuadcastRGB: {exc}") from exc
        self._process = process
        time.sleep(0.1)
        return_code = process.poll()
        if return_code is not None:
            details = (
                process.stderr.read().decode("utf-8", errors="replace").strip()
                if process.stderr is not None
                else ""
            )
            self._process = None
            raise RuntimeError(
                f"QuadcastRGB exited with status {return_code}"
                + (f": {details}" if details else "")
            )
        self._mode = arguments

    def _stop(self) -> None:
        process = self._process
        self._process = None
        self._mode = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stderr is not None:
            process.stderr.close()


class CombinedIndicator:
    def __init__(self, indicators: Sequence[Indicator]) -> None:
        self._indicators = tuple(indicators)

    def start_warmup(self) -> None:
        for indicator in self._indicators:
            indicator.start_warmup()

    def finish_warmup(self) -> None:
        for indicator in self._indicators:
            indicator.finish_warmup()

    def turn_on(self) -> None:
        for indicator in self._indicators:
            indicator.turn_on()

    def speaking(self) -> None:
        for indicator in self._indicators:
            indicator.speaking()

    def turn_off(self) -> None:
        for indicator in self._indicators:
            indicator.turn_off()

    def close(self) -> None:
        for indicator in self._indicators:
            indicator.close()


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


class SpeechPlayer:
    """Speak execution responses through the configured audio output."""

    def __init__(
        self,
        *,
        model_path: Path,
        device: str | None,
        alsa_executable: str = "aplay",
        pipewire_executable: str = "pw-play",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.alsa_executable = alsa_executable
        self.pipewire_executable = pipewire_executable
        self._speech_lock = Lock()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Piper voice model does not exist: {model_path}")
        if not Path(f"{self.model_path}.json").is_file():
            raise FileNotFoundError(
                f"Piper voice configuration does not exist: {self.model_path}.json"
            )
        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise RuntimeError("piper-tts is required for speech output") from exc

        self._voice: TtsVoice = PiperVoice.load(self.model_path)

    def warmup(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            self._synthesize_to_file(PIPER_WARMUP_TEXT, audio_file.name)

    def speak(self, text: str) -> None:
        with self._speech_lock:
            self._speak(text)

    def _speak(self, text: str) -> None:
        if not text.strip():
            raise ValueError("speech text must not be empty")
        pipewire_target: str | None = None
        if self.device is not None and self.device.startswith("pipewire:"):
            pipewire_target = self.device.removeprefix("pipewire:") or "auto"
            playback_name = self.pipewire_executable
        else:
            playback_name = self.alsa_executable
        playback = shutil.which(playback_name)
        if playback is None:
            raise RuntimeError(
                f"{playback_name!r} was not found; install the required audio tools"
            )

        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            synthesis_started = time.perf_counter()
            try:
                self._synthesize_to_file(text, audio_file.name)
            finally:
                logging.getLogger(__name__).info(
                    "speech synthesis completed: duration_ms=%.1f",
                    (time.perf_counter() - synthesis_started) * 1_000,
                )

            if pipewire_target is not None:
                command = [
                    playback,
                    "--target",
                    pipewire_target,
                    audio_file.name,
                ]
            else:
                command = [playback, "--quiet"]
                if self.device:
                    command.extend(["--device", self.device])
                command.append(audio_file.name)
            playback_started = time.perf_counter()
            try:
                result = subprocess.run(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=15,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("speech playback timed out") from exc
            finally:
                logging.getLogger(__name__).info(
                    "speech playback completed: duration_ms=%.1f",
                    (time.perf_counter() - playback_started) * 1_000,
                )
            if result.returncode != 0:
                details = result.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"speech playback failed with status {result.returncode}"
                    + (f": {details}" if details else "")
                )

    def _synthesize_to_file(self, text: str, path: str) -> None:
        try:
            with wave.open(path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self._voice.config.sample_rate)
                self._voice.synthesize_wav(text, wav_file)
        except (OSError, RuntimeError, ValueError, wave.Error) as exc:
            raise RuntimeError(f"Piper speech generation failed: {exc}") from exc


def speak_with_indicator(
    speech_player: SpeechPlayer,
    indicator: Indicator | None,
    text: str,
) -> None:
    if indicator is not None:
        indicator.speaking()
    try:
        speech_player.speak(text)
    finally:
        if indicator is not None:
            indicator.turn_off()


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


def record_command(capture: AudioCapture) -> np.ndarray | None:
    """Record one utterance, ending after sustained silence or the duration cap."""
    start_timeout_frames = math.ceil(
        COMMAND_START_TIMEOUT_SECONDS / (FRAME_MILLISECONDS / 1_000)
    )
    max_command_frames = math.ceil(
        COMMAND_MAX_DURATION_SECONDS / (FRAME_MILLISECONDS / 1_000)
    )
    end_silence_frames = math.ceil(
        COMMAND_END_SILENCE_SECONDS / (FRAME_MILLISECONDS / 1_000)
    )
    minimum_voiced_frames = math.ceil(
        COMMAND_MIN_VOICED_SECONDS / (FRAME_MILLISECONDS / 1_000)
    )
    recorded_frames: list[bytes] = []
    speech_started = False
    command_frames = 0
    voiced_frames = 0
    silence_frames = 0

    for frame_index in range(start_timeout_frames + max_command_frames):
        frame = capture.read_frame()
        if frame is None:
            raise RuntimeError("audio capture ended before command recording completed")
        if len(frame) != FRAME_BYTES:
            raise ValueError(
                f"command audio frame must be exactly {FRAME_BYTES} bytes"
            )
        samples = np.frombuffer(frame, dtype="<i2").astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(samples / 32768.0))))
        level_dbfs = 20.0 * math.log10(max(rms, 1e-12))
        voiced = level_dbfs > COMMAND_VOICE_THRESHOLD_DBFS

        if not speech_started:
            if voiced:
                speech_started = True
            elif frame_index + 1 >= start_timeout_frames:
                return None
            else:
                continue

        recorded_frames.append(frame)
        command_frames += 1
        if voiced:
            voiced_frames += 1
            silence_frames = 0
        else:
            silence_frames += 1
            if silence_frames >= end_silence_frames:
                del recorded_frames[-silence_frames:]
                break
        if command_frames >= max_command_frames:
            break

    if voiced_frames < minimum_voiced_frames:
        return None
    return np.frombuffer(b"".join(recorded_frames), dtype="<i2").copy()


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
        description=(
            "Listen for Alexa, acknowledge detection, and optionally run "
            "ME2 command inference."
        )
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
    parser.add_argument(
        "--hyperx-rgb-executable",
        type=Path,
        help="enable HyperX DuoCast RGB status lighting using QuadcastRGB",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--intent-model",
        type=Path,
        help="ME2 probability-output ONNX model; enables command inference",
    )
    parser.add_argument(
        "--mel-stats",
        type=Path,
        help="training-only mean/std NPZ paired with --intent-model",
    )
    parser.add_argument(
        "--intent-threshold",
        type=float,
        help="explicit in-scope confidence threshold paired with --intent-model",
    )
    parser.add_argument(
        "--force-intent",
        choices=EXECUTION_INTENT_NAMES,
        help=(
            "dispatch this intent for the next spoken command only, without "
            "loading an ME2 model"
        ),
    )
    parser.add_argument(
        "--dummy-time",
        action="store_true",
        help=(
            "persistently dispatch TIME for every voiced command after Alexa; "
            "bypasses real ME2 inference"
        ),
    )
    parser.add_argument(
        "--dummy-intent",
        choices=EXECUTION_INTENT_NAMES,
        help=(
            "persistently dispatch this intent for every voiced command after "
            "Alexa, without interpreting the words or loading ME2"
        ),
    )
    parser.add_argument(
        "--tts-model",
        type=Path,
        default=Path(__file__).resolve().parent
        / "models"
        / "tts"
        / "en_US-lessac-medium.onnx",
        help="Piper ONNX voice model used to speak intent execution results",
    )
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
    intent_configuration = (
        args.intent_model,
        args.mel_stats,
        args.intent_threshold,
    )
    if any(value is not None for value in intent_configuration) and not all(
        value is not None for value in intent_configuration
    ):
        parser.error(
            "--intent-model, --mel-stats, and --intent-threshold must be "
            "provided together"
        )
    if args.force_intent is not None and args.intent_model is not None:
        parser.error("--force-intent cannot be combined with ME2 model options")
    if args.dummy_time and args.intent_model is not None:
        parser.error("--dummy-time cannot be combined with ME2 model options")
    if args.dummy_time and args.force_intent is not None:
        parser.error("--dummy-time cannot be combined with --force-intent")
    if args.dummy_intent is not None and args.intent_model is not None:
        parser.error("--dummy-intent cannot be combined with ME2 model options")
    if args.dummy_intent is not None and (
        args.dummy_time or args.force_intent is not None
    ):
        parser.error("--dummy-intent cannot be combined with other forced modes")
    if args.intent_threshold is not None and (
        not math.isfinite(args.intent_threshold)
        or not 0.0 <= args.intent_threshold <= 1.0
    ):
        parser.error("--intent-threshold must be finite and in [0, 1]")
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
    indicator: Indicator | None = None
    dispatcher: IntentDispatcher | None = None
    speech_player: SpeechPlayer | None = None
    try:
        indicators: list[Indicator] = []
        if not args.no_led:
            indicators.append(GpioIndicator(args.led_gpio))
        hyperx_executable = getattr(args, "hyperx_rgb_executable", None)
        if hyperx_executable is not None:
            indicators.append(HyperxRgbIndicator(hyperx_executable))
        indicator = CombinedIndicator(indicators) if indicators else None
        if indicator is not None:
            indicator.start_warmup()

        try:
            detector = OpenWakeWordDetector(args.wakeword_model)
            warmup_started = time.monotonic()
            detector.warmup()
            logger.info(
                "wake-word model warmup complete: frames=%d duration_ms=%.1f",
                WAKE_MODEL_WARMUP_FRAMES,
                (time.monotonic() - warmup_started) * 1_000,
            )
            beep_player = AudioBeepPlayer(device=args.output_device)
            intent_model = getattr(args, "intent_model", None)
            mel_stats = getattr(args, "mel_stats", None)
            intent_threshold = getattr(args, "intent_threshold", None)
            predictor = (
                OnnxIntentPredictor(intent_model, mel_stats)
                if intent_model is not None
                else None
            )
            forced_intent = getattr(args, "force_intent", None)
            dummy_time = getattr(args, "dummy_time", False)
            dummy_intent = getattr(args, "dummy_intent", None)
            if (
                predictor is not None
                or forced_intent is not None
                or dummy_time
                or dummy_intent is not None
            ):
                logger.info("loading Piper voice model for low-latency speech")
                initialized_speech_player = SpeechPlayer(
                    model_path=args.tts_model,
                    device=args.output_device,
                )
                warmup_started = time.monotonic()
                initialized_speech_player.warmup()
                logger.info(
                    "Piper voice model ready: silent synthesis warmup_ms=%.1f",
                    (time.monotonic() - warmup_started) * 1_000,
                )
                speech_player = initialized_speech_player
                dispatcher = IntentDispatcher(
                    timer_expiry_callback=lambda: initialized_speech_player.speak(
                        TIMER_EXPIRED_MESSAGE
                    ),
                    alarm_expiry_callback=initialized_speech_player.speak,
                )
        finally:
            if indicator is not None:
                indicator.finish_warmup()

        logger.info(
            "wake listener started: wake_word=alexa input=%s output=%s "
            "threshold=%.3f continuous=%s intent_inference=%s forced_intent=%s "
            "dummy_time=%s dummy_intent=%s",
            args.input_device,
            args.output_device or "default",
            args.threshold,
            not args.once,
            predictor is not None,
            forced_intent or "disabled",
            dummy_time,
            dummy_intent or "disabled",
        )
        if dummy_time:
            logger.warning(
                "DUMMY MODE: forcing intent=TIME for every voiced command; "
                "real ONNX inference is disabled"
            )
        elif dummy_intent is not None:
            logger.warning(
                "DUMMY MODE: forcing intent=%s for every voiced command; "
                "real ONNX inference is disabled",
                dummy_intent,
            )
        while True:
            capture.start()
            score = listen_for_wake(detector, capture, threshold=args.threshold)
            acknowledge_wake(capture, indicator, beep_player)
            logger.info(
                "wake acknowledged: wake_word=alexa score=%.4f",
                score,
            )
            interaction_started = time.perf_counter()
            indicator_hold_started = time.monotonic()
            command: np.ndarray | None = None
            active_forced_intent = forced_intent
            if (
                (
                    predictor is not None
                    or active_forced_intent is not None
                    or dummy_time
                    or dummy_intent is not None
                )
                and dispatcher is not None
            ):
                capture.start()
                capture_started = time.perf_counter()
                try:
                    command = record_command(capture)
                finally:
                    capture.close()
                    logger.info(
                        "command capture completed: duration_ms=%.1f "
                        "speech_detected=%s",
                        (time.perf_counter() - capture_started) * 1_000,
                        command is not None,
                    )
            if predictor is not None and dispatcher is not None:
                if command is None:
                    logger.warning(
                        "no command speech detected within %.1f seconds",
                        COMMAND_START_TIMEOUT_SECONDS,
                    )
                else:
                    prediction = predictor.predict(
                        command, threshold=intent_threshold
                    )
                    logger.info(
                        "intent prediction: label=%s confidence=%.4f "
                        "out_of_scope_confidence=%.4f accepted=%s",
                        prediction.intent_label,
                        prediction.confidence,
                        prediction.out_of_scope_confidence,
                        prediction.accepted,
                    )
                    dispatch_label = (
                        prediction.intent_label
                        if prediction.accepted
                        else "OUT_OF_SCOPE"
                    )
                    execution_started = time.perf_counter()
                    if dispatch_label == "CREATE_REMINDER":
                        reminder_action = CREATE_REMINDER_ACTIONS[
                            prediction.slot_ids[CREATE_REMINDER_SLOT_INDEX]
                        ]
                        result = dispatcher.execute(
                            dispatch_label,
                            slot_action=reminder_action,
                        )
                    elif dispatch_label == "TIMER":
                        timer_slot = prediction.slot_ids[TIMER_SLOT_INDEX]
                        if not 0 <= timer_slot < len(TIMER_ACTIONS):
                            raise RuntimeError(
                                f"invalid TIMER slot prediction: {timer_slot}"
                            )
                        result = dispatcher.execute(
                            dispatch_label,
                            slot_action=TIMER_ACTIONS[timer_slot],
                        )
                    elif dispatch_label == "ALARM":
                        alarm_slot = prediction.slot_ids[ALARM_SLOT_INDEX]
                        if not 0 <= alarm_slot < len(ALARM_ACTIONS):
                            raise RuntimeError(
                                f"invalid ALARM slot prediction: {alarm_slot}"
                            )
                        result = dispatcher.execute(
                            dispatch_label,
                            slot_action=ALARM_ACTIONS[alarm_slot],
                        )
                    else:
                        result = dispatcher.execute(dispatch_label)
                    logger.info(
                        "intent execution completed: intent=%s duration_ms=%.1f",
                        dispatch_label,
                        (time.perf_counter() - execution_started) * 1_000,
                    )
                    logger.info(
                        "intent result: intent=%s status=%s message=%s",
                        result.intent,
                        result.status,
                        result.message,
                    )
                    if speech_player is not None:
                        speak_with_indicator(speech_player, indicator, result.message)
            elif (
                (
                    active_forced_intent is not None
                    or dummy_time
                    or dummy_intent is not None
                )
                and dispatcher is not None
            ):
                if dummy_time:
                    intent_to_execute = "TIME"
                elif dummy_intent is not None:
                    intent_to_execute = dummy_intent
                elif active_forced_intent is not None:
                    intent_to_execute = active_forced_intent
                else:
                    raise RuntimeError("forced intent disappeared before dispatch")
                if command is None:
                    logger.warning(
                        "%s intent %s was not executed because no command speech "
                        "was detected",
                        "dummy"
                        if dummy_time or dummy_intent is not None
                        else "forced",
                        intent_to_execute,
                    )
                else:
                    logger.info(
                        "%s intent dispatch: intent=%s",
                        "dummy"
                        if dummy_time or dummy_intent is not None
                        else "forced",
                        intent_to_execute,
                    )
                    execution_started = time.perf_counter()
                    result = dispatcher.execute(intent_to_execute)
                    logger.info(
                        "intent result: intent=%s status=%s message=%s "
                        "inference_mode=%s",
                        result.intent,
                        result.status,
                        result.message,
                        "dummy_intent"
                        if dummy_intent is not None
                        else "dummy_time"
                        if dummy_time
                        else "forced",
                    )
                    logger.info(
                        "intent execution completed: intent=%s duration_ms=%.1f",
                        intent_to_execute,
                        (time.perf_counter() - execution_started) * 1_000,
                    )
                    if speech_player is not None:
                        speak_with_indicator(speech_player, indicator, result.message)
                if not dummy_time and dummy_intent is None:
                    forced_intent = None
                    logger.info("forced intent cleared; returning to wake standby")
            logger.info(
                "wake-to-reply interaction completed: duration_ms=%.1f",
                (time.perf_counter() - interaction_started) * 1_000,
            )
            if indicator is not None:
                remaining_hold = args.indicator_hold_seconds - (
                    time.monotonic() - indicator_hold_started
                )
                if remaining_hold > 0:
                    time.sleep(remaining_hold)
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
        if dispatcher is not None:
            dispatcher.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logging.getLogger("pi5_vcm.inference").exception(
            "wake acknowledgement runtime failed: %s",
            exc,
        )
        raise SystemExit(1) from exc
