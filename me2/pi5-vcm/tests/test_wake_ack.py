from __future__ import annotations

import io
import struct
import sys
import wave
from argparse import Namespace
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inference.pi5_runner import wake_ack as runtime
from inference.pi5_runner.wake_ack import (
    FRAME_BYTES,
    FRAME_SAMPLES,
    AlsaCapture,
    AudioBeepPlayer,
    OpenWakeWordDetector,
    SpeechPlayer,
    acknowledge_wake,
    listen_for_wake,
    make_beep_pcm,
    record_command,
    read_exact,
)


def test_read_exact_handles_partial_reads() -> None:
    class PartialReader(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            return super().read(min(size, 3))

    assert read_exact(PartialReader(b"abcdefgh"), 8) == b"abcdefgh"


def test_read_exact_returns_none_only_for_clean_eof() -> None:
    assert read_exact(io.BytesIO(b""), 4) is None
    with pytest.raises(RuntimeError, match="middle of a frame"):
        read_exact(io.BytesIO(b"abc"), 4)


class FakeWakeWordModel:
    def __init__(self, scores: list[float]) -> None:
        self.scores = deque(scores)
        self.inputs: list[np.ndarray] = []
        self.reset_count = 0

    def predict(self, audio: np.ndarray) -> dict[str, float]:
        self.inputs.append(audio.copy())
        return {"alexa": self.scores.popleft()}

    def reset(self) -> None:
        self.reset_count += 1


def test_detector_passes_16khz_pcm_to_wakeword_model(tmp_path: Path) -> None:
    model = FakeWakeWordModel([0.72])
    detector = OpenWakeWordDetector(tmp_path / "unused.onnx", model=model)
    frame = np.arange(FRAME_BYTES // 2, dtype="<i2").tobytes()

    assert detector.score_frame(frame) == pytest.approx(0.72)
    np.testing.assert_array_equal(
        model.inputs[0], np.frombuffer(frame, dtype="<i2")
    )


def test_detector_reset_clears_model_state(tmp_path: Path) -> None:
    model = FakeWakeWordModel([0.72])
    detector = OpenWakeWordDetector(tmp_path / "unused.onnx", model=model)

    detector.reset()

    assert model.reset_count == 1


def test_detector_warmup_uses_silence_and_resets_model(tmp_path: Path) -> None:
    model = FakeWakeWordModel([0.0] * runtime.WAKE_MODEL_WARMUP_FRAMES)
    detector = OpenWakeWordDetector(tmp_path / "unused.onnx", model=model)

    detector.warmup()

    assert len(model.inputs) == runtime.WAKE_MODEL_WARMUP_FRAMES
    assert all(
        np.array_equal(frame, np.zeros(FRAME_SAMPLES, dtype=np.int16))
        for frame in model.inputs
    )
    assert model.reset_count == 1


def test_hyperx_rgb_indicator_runs_warmup_wake_and_off_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "quadcastrgb"
    executable.touch()
    executable.chmod(0o755)
    commands: list[list[str]] = []
    stopped: list[bool] = []

    class Process:
        def __init__(self) -> None:
            self.running = True
            self.stderr = io.BytesIO()

        def poll(self) -> int | None:
            return None if self.running else 0

        def terminate(self) -> None:
            self.running = False
            stopped.append(True)

        def wait(self, timeout: float | None = None) -> int:
            self.running = False
            return 0

        def kill(self) -> None:
            self.running = False

    def fake_popen(command: list[str], **kwargs: object) -> Process:
        commands.append(command)
        return Process()

    monkeypatch.setattr(
        "inference.pi5_runner.wake_ack.subprocess.run",
        lambda *args, **kwargs: pytest.fail("RGB mode must stay in foreground"),
    )
    monkeypatch.setattr(
        "inference.pi5_runner.wake_ack.subprocess.Popen",
        fake_popen,
    )
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)

    color_store = runtime.InteractionColorStore(
        tmp_path / "interaction-color.json"
    )
    indicator = runtime.HyperxRgbIndicator(
        executable,
        interaction_color=color_store.load(),
    )
    indicator.start_warmup()
    indicator.finish_warmup()
    indicator.turn_on()
    indicator.speaking()
    dispatcher = runtime.IntentDispatcher(
        interaction_color_store=color_store,
        interaction_color_callback=indicator.set_interaction_color,
    )
    result = dispatcher.execute("COLOR", slot_action="Red")
    assert result.message == "Interaction color set to red."
    assert runtime.InteractionColorStore(color_store.path).load() == "Red"
    indicator.turn_on()
    indicator.speaking()
    indicator.turn_off()
    dispatcher.close()
    indicator.close()

    assert commands == [
        [str(executable), "-a", "cycle"],
        [str(executable), "-a", "solid", "0"],
        [str(executable), "-a", "solid", "0000ff", "-b", "100"],
        [str(executable), "-a", "pulse", "0000ff", "-b", "100"],
        [str(executable), "-a", "solid", "ff0000", "-b", "100"],
        [str(executable), "-a", "pulse", "ff0000", "-b", "100"],
        [str(executable), "-a", "solid", "0"],
    ]
    assert len(stopped) == 7


def test_hyperx_rgb_indicator_applies_brightness_to_current_and_future_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "quadcastrgb"
    executable.touch()
    executable.chmod(0o755)
    commands: list[list[str]] = []

    class Process:
        def __init__(self) -> None:
            self.running = True
            self.stderr = io.BytesIO()

        def poll(self) -> int | None:
            return None if self.running else 0

        def terminate(self) -> None:
            self.running = False

        def wait(self, timeout: float | None = None) -> int:
            self.running = False
            return 0

        def kill(self) -> None:
            self.running = False

    monkeypatch.setattr(
        "inference.pi5_runner.wake_ack.subprocess.Popen",
        lambda command, **kwargs: commands.append(command) or Process(),
    )
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)

    indicator = runtime.HyperxRgbIndicator(executable)
    indicator.turn_on()
    indicator.set_brightness(60)
    indicator.speaking()
    indicator.turn_off()
    indicator.close()

    assert commands == [
        [str(executable), "-a", "solid", "0000ff", "-b", "100"],
        [str(executable), "-a", "solid", "0000ff", "-b", "60"],
        [str(executable), "-a", "pulse", "0000ff", "-b", "60"],
        [str(executable), "-a", "solid", "0"],
    ]


def test_main_rearms_after_acknowledgement_until_interrupted(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    caplog.set_level("INFO")
    events: list[str] = []

    class Capture:
        def start(self) -> None:
            events.append("capture-start")

        def close(self) -> None:
            events.append("capture-close")

    class Detector:
        def __init__(self, model_path: Path) -> None:
            pass

        def warmup(self) -> None:
            events.append("detector-warmup")

        def reset(self) -> None:
            events.append("detector-reset")
            raise KeyboardInterrupt

    args = Namespace(
        input_device="test-mic",
        indicator_hold_seconds=0.0,
        led_gpio=None,
        no_led=True,
        once=False,
        output_device="pipewire:auto",
        threshold=0.5,
        wakeword_model=tmp_path / "alexa.onnx",
    )
    monkeypatch.setattr(runtime, "_parse_args", lambda argv=None: args)
    monkeypatch.setattr(runtime, "AlsaCapture", lambda **kwargs: Capture())
    monkeypatch.setattr(runtime, "OpenWakeWordDetector", Detector)
    monkeypatch.setattr(runtime, "AudioBeepPlayer", lambda **kwargs: object())
    monkeypatch.setattr(
        runtime,
        "listen_for_wake",
        lambda *args, **kwargs: 0.82,
    )
    monkeypatch.setattr(
        runtime,
        "acknowledge_wake",
        lambda capture, indicator, beep: events.append("wake-acknowledged"),
    )

    assert runtime.main() == 0

    assert events == [
        "detector-warmup",
        "capture-start",
        "wake-acknowledged",
        "detector-reset",
        "capture-close",
    ]
    assert "wake acknowledged: wake_word=alexa score=0.8200" in caplog.text


def test_detector_rejects_invalid_frame_size(tmp_path: Path) -> None:
    detector = OpenWakeWordDetector(
        tmp_path / "unused.onnx", model=FakeWakeWordModel([0.1])
    )

    with pytest.raises(ValueError, match="exactly"):
        detector.score_frame(b"short")


def test_detector_rejects_unsupported_model_file(tmp_path: Path) -> None:
    model_path = tmp_path / "alexa.pt"
    model_path.touch()

    with pytest.raises(ValueError, match="ONNX or TFLite"):
        OpenWakeWordDetector(model_path)


def test_detector_rejects_invalid_model_score(tmp_path: Path) -> None:
    detector = OpenWakeWordDetector(
        tmp_path / "unused.onnx", model=FakeWakeWordModel([1.2])
    )

    with pytest.raises(RuntimeError, match="invalid score"):
        detector.score_frame(bytes(FRAME_BYTES))


def test_parse_args_accepts_complete_intent_configuration() -> None:
    args = runtime._parse_args(
        [
            "--wakeword-model",
            "alexa.onnx",
            "--no-led",
            "--intent-model",
            "me2.onnx",
            "--mel-stats",
            "mel_stats.npz",
            "--intent-threshold",
            "0.78",
        ]
    )

    assert args.intent_model == Path("me2.onnx")
    assert args.mel_stats == Path("mel_stats.npz")
    assert args.intent_threshold == pytest.approx(0.78)


def test_parse_args_defaults_to_system_audio_devices() -> None:
    args = runtime._parse_args(
        ["--wakeword-model", "alexa.onnx", "--no-led"]
    )

    assert args.input_device == "default"
    assert args.output_device is None


def test_parse_args_requires_paired_intent_configuration() -> None:
    with pytest.raises(SystemExit):
        runtime._parse_args(
            [
                "--wakeword-model",
                "alexa.onnx",
                "--no-led",
                "--intent-model",
                "me2.onnx",
            ]
        )


def test_parse_args_accepts_one_wake_forced_intent() -> None:
    args = runtime._parse_args(
        [
            "--wakeword-model",
            "alexa.onnx",
            "--no-led",
            "--force-intent",
            "TIME",
        ]
    )

    assert args.force_intent == "TIME"


def test_parse_args_accepts_persistent_dummy_time() -> None:
    args = runtime._parse_args(
        [
            "--wakeword-model",
            "alexa.onnx",
            "--no-led",
            "--dummy-time",
        ]
    )

    assert args.dummy_time


def test_parse_args_accepts_persistent_dummy_intent() -> None:
    args = runtime._parse_args(
        [
            "--wakeword-model",
            "alexa.onnx",
            "--no-led",
            "--dummy-intent",
            "WEATHER",
        ]
    )

    assert args.dummy_intent == "WEATHER"


def test_parse_args_accepts_hyperx_rgb_executable() -> None:
    args = runtime._parse_args(
        [
            "--wakeword-model",
            "alexa.onnx",
            "--no-led",
            "--hyperx-rgb-executable",
            "/home/user/.local/bin/quadcastrgb",
        ]
    )

    assert args.hyperx_rgb_executable == Path(
        "/home/user/.local/bin/quadcastrgb"
    )


def test_parse_args_rejects_forced_intent_with_model_configuration() -> None:
    with pytest.raises(SystemExit):
        runtime._parse_args(
            [
                "--wakeword-model",
                "alexa.onnx",
                "--no-led",
                "--force-intent",
                "TIME",
                "--intent-model",
                "me2.onnx",
                "--mel-stats",
                "mel_stats.npz",
                "--intent-threshold",
                "0.78",
            ]
        )


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--intent-model", "me2.onnx", "--mel-stats", "mel_stats.npz",
         "--intent-threshold", "0.78"],
        ["--force-intent", "TIME"],
        ["--dummy-intent", "WEATHER"],
    ],
)
def test_parse_args_rejects_dummy_time_with_other_inference_modes(
    extra_args: list[str],
) -> None:
    with pytest.raises(SystemExit):
        runtime._parse_args(
            [
                "--wakeword-model",
                "alexa.onnx",
                "--no-led",
                "--dummy-time",
                *extra_args,
            ]
        )


def test_forced_intent_applies_to_one_wake_then_clears(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    caplog.set_level("INFO")
    events: list[str] = []
    wake_count = 0
    record_count = 0

    class Capture:
        def start(self) -> None:
            events.append("capture-start")

        def close(self) -> None:
            events.append("capture-close")

    class Detector:
        def __init__(self, model_path: Path) -> None:
            pass

        def warmup(self) -> None:
            events.append("detector-warmup")

        def reset(self) -> None:
            events.append("detector-reset")
            if events.count("detector-reset") == 2:
                raise KeyboardInterrupt

    class Dispatcher:
        def __init__(self, **kwargs) -> None:
            pass

        def execute(self, intent: str) -> object:
            events.append(f"dispatch:{intent}")
            return type(
                "Result",
                (),
                {"intent": intent, "status": "completed", "message": "test"},
            )()

        def wait_for_call_end(self) -> None:
            events.append("wait-for-call-end")

        def close(self) -> None:
            events.append("dispatcher-close")

    args = Namespace(
        force_intent="TIME",
        input_device="test-mic",
        indicator_hold_seconds=0.0,
        led_gpio=None,
        no_led=True,
        once=False,
        output_device="pipewire:auto",
        threshold=0.5,
        tts_model=tmp_path / "voice.onnx",
        wakeword_model=tmp_path / "alexa.onnx",
    )
    monkeypatch.setattr(runtime, "_parse_args", lambda argv=None: args)
    monkeypatch.setattr(runtime, "AlsaCapture", lambda **kwargs: Capture())
    monkeypatch.setattr(runtime, "OpenWakeWordDetector", Detector)
    monkeypatch.setattr(runtime, "AudioBeepPlayer", lambda **kwargs: object())
    monkeypatch.setattr(runtime, "IntentDispatcher", Dispatcher)
    monkeypatch.setattr(
        runtime,
        "SpeechPlayer",
        lambda **kwargs: SimpleNamespace(
            warmup=lambda: events.append("tts-warmup"),
            speak=lambda text: events.append("speak"),
        ),
    )

    def listen(*args: object, **kwargs: object) -> float:
        nonlocal wake_count
        wake_count += 1
        return 0.9

    def record(capture: Capture) -> np.ndarray | None:
        nonlocal record_count
        record_count += 1
        events.append("record-command")
        return np.zeros(10, dtype=np.int16)

    monkeypatch.setattr(runtime, "listen_for_wake", listen)
    monkeypatch.setattr(runtime, "acknowledge_wake", lambda *a, **kw: None)
    monkeypatch.setattr(runtime, "record_command", record)

    assert runtime.main() == 0

    assert wake_count == 2
    assert record_count == 1
    assert events.index("wait-for-call-end") < events.index("capture-start")
    assert events.count("dispatch:TIME") == 1
    assert events.count("speak") == 1
    assert events.index("dispatch:TIME") < events.index("detector-reset")
    assert "forced intent cleared; returning to wake standby" in caplog.text


@pytest.mark.parametrize(
    (
        "dummy_time",
        "dummy_intent",
        "expected_intent",
        "expected_mode",
        "expected_warning",
    ),
    [
        (True, None, "TIME", "dummy_time", "DUMMY MODE: forcing intent=TIME"),
        (
            False,
            "WEATHER",
            "WEATHER",
            "dummy_intent",
            "DUMMY MODE: forcing intent=WEATHER",
        ),
    ],
)
def test_persistent_dummy_intent_dispatches_and_speaks_on_every_wake(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    dummy_time: bool,
    dummy_intent: str | None,
    expected_intent: str,
    expected_mode: str,
    expected_warning: str,
) -> None:
    caplog.set_level("INFO")
    events: list[str] = []

    class Capture:
        def start(self) -> None:
            events.append("capture-start")

        def close(self) -> None:
            events.append("capture-close")

    class Detector:
        def __init__(self, model_path: Path) -> None:
            pass

        def warmup(self) -> None:
            events.append("detector-warmup")

        def reset(self) -> None:
            events.append("detector-reset")
            if events.count("detector-reset") == 3:
                raise KeyboardInterrupt

    class Dispatcher:
        def __init__(self, **kwargs) -> None:
            pass

        def execute(self, intent: str) -> object:
            events.append(f"dispatch:{intent}")
            return type(
                "Result",
                (),
                {
                    "intent": intent,
                    "status": "completed",
                    "message": "The local time is test.",
                },
            )()

        def wait_for_call_end(self) -> None:
            events.append("wait-for-call-end")

        def close(self) -> None:
            events.append("dispatcher-close")

    args = Namespace(
        dummy_intent=dummy_intent,
        dummy_time=dummy_time,
        input_device="test-mic",
        indicator_hold_seconds=0.0,
        led_gpio=None,
        no_led=True,
        once=False,
        output_device="pipewire:auto",
        threshold=0.5,
        tts_model=tmp_path / "voice.onnx",
        wakeword_model=tmp_path / "alexa.onnx",
    )
    monkeypatch.setattr(runtime, "_parse_args", lambda argv=None: args)
    monkeypatch.setattr(runtime, "AlsaCapture", lambda **kwargs: Capture())
    monkeypatch.setattr(runtime, "OpenWakeWordDetector", Detector)
    monkeypatch.setattr(runtime, "AudioBeepPlayer", lambda **kwargs: object())
    monkeypatch.setattr(runtime, "IntentDispatcher", Dispatcher)
    monkeypatch.setattr(
        runtime,
        "OnnxIntentPredictor",
        lambda *args, **kwargs: pytest.fail("dummy mode must bypass ONNX"),
    )
    monkeypatch.setattr(
        runtime,
        "SpeechPlayer",
        lambda **kwargs: type(
            "Speech",
            (),
            {
                "warmup": lambda self: events.append("tts-warmup"),
                "speak": lambda self, text: events.append(f"speak:{text}"),
            },
        )(),
    )
    wake_count = 0
    command_results: deque[np.ndarray | None] = deque(
        [None, np.ones(100, dtype=np.int16), np.ones(100, dtype=np.int16)]
    )

    def listen(*args: object, **kwargs: object) -> float:
        nonlocal wake_count
        wake_count += 1
        return 0.9

    def record(capture: Capture) -> np.ndarray:
        events.append("record-command")
        command = command_results.popleft()
        if command is None:
            return None
        return command

    monkeypatch.setattr(runtime, "listen_for_wake", listen)
    monkeypatch.setattr(runtime, "acknowledge_wake", lambda *a, **kw: None)
    monkeypatch.setattr(runtime, "record_command", record)

    assert runtime.main() == 0

    assert wake_count == 3
    assert events.count("record-command") == 3
    assert events.count(f"dispatch:{expected_intent}") == 2
    assert events.count("speak:The local time is test.") == 2
    assert expected_warning in caplog.text
    assert f"inference_mode={expected_mode}" in caplog.text
    assert "forced intent cleared" not in caplog.text
    assert f"dummy intent {expected_intent} was not executed" in caplog.text
    assert "dispatcher-close" in events


class FakeCapture:
    def __init__(self, frames: list[bytes]) -> None:
        self.frames = deque(frames)
        self.closed = False

    def read_frame(self) -> bytes | None:
        return self.frames.popleft()

    def close(self) -> None:
        self.closed = True


def test_listener_waits_until_score_reaches_threshold(tmp_path: Path) -> None:
    model = FakeWakeWordModel([0.2, 0.5])
    detector = OpenWakeWordDetector(tmp_path / "unused.onnx", model=model)
    frame = bytes(FRAME_BYTES)

    score = listen_for_wake(
        detector,
        FakeCapture([frame, frame]),
        threshold=0.5,
    )

    assert score == pytest.approx(0.5)
    assert len(model.inputs) == 2


def test_acknowledgement_stops_capture_before_indicator_and_beep() -> None:
    events: list[str] = []

    class Capture:
        def read_frame(self) -> bytes | None:
            return None

        def close(self) -> None:
            events.append("capture-stopped")

    class Indicator:
        def turn_on(self) -> None:
            events.append("indicator-on")

        def speaking(self) -> None:
            events.append("indicator-speaking")

        def turn_off(self) -> None:
            events.append("indicator-off")

        def close(self) -> None:
            pass

    class Beep:
        def play(self) -> None:
            events.append("beep")

    acknowledge_wake(Capture(), Indicator(), Beep())

    assert events == ["capture-stopped", "indicator-on", "beep"]


def test_speech_lights_indicator_green_and_turns_it_off_afterward() -> None:
    events: list[str] = []

    class Indicator:
        def speaking(self) -> None:
            events.append("green")

        def turn_off(self) -> None:
            events.append("off")

    class SpeechPlayer:
        def speak(self, text: str) -> None:
            events.append(f"speaking:{text}")

    runtime.speak_with_indicator(SpeechPlayer(), Indicator(), "The time is now.")

    assert events == ["green", "speaking:The time is now.", "off"]


def test_speech_turns_indicator_off_when_playback_fails() -> None:
    events: list[str] = []

    class Indicator:
        def speaking(self) -> None:
            events.append("green")

        def turn_off(self) -> None:
            events.append("off")

    class SpeechPlayer:
        def speak(self, text: str) -> None:
            raise RuntimeError("playback failed")

    with pytest.raises(RuntimeError, match="playback failed"):
        runtime.speak_with_indicator(SpeechPlayer(), Indicator(), "Hello.")

    assert events == ["green", "off"]


def test_beep_pcm_has_expected_duration_and_fades() -> None:
    pcm = make_beep_pcm(sample_rate=1_000, frequency_hz=100, duration_seconds=0.1)
    samples = np.frombuffer(pcm, dtype="<i2")

    assert len(samples) == 100
    assert samples[0] == 0
    assert samples[-1] == 0
    assert np.max(np.abs(samples)) > 0


def test_beep_player_surfaces_alsa_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        returncode = 1
        stderr = b"no playback device"

    monkeypatch.setattr(
        "inference.pi5_runner.wake_ack.shutil.which",
        lambda _: "/usr/bin/aplay",
    )
    monkeypatch.setattr(
        "inference.pi5_runner.wake_ack.subprocess.run",
        lambda *args, **kwargs: Result(),
    )

    with pytest.raises(RuntimeError, match="no playback device"):
        AudioBeepPlayer(device=None).play()


def test_alsa_capture_uses_supported_file_type_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        stdout = None

        def poll(self) -> int:
            return 0

    commands: list[list[str]] = []

    def fake_popen(command: list[str], **kwargs: object) -> Process:
        commands.append(command)
        return Process()

    monkeypatch.setattr(
        "inference.pi5_runner.wake_ack.shutil.which",
        lambda _: "/usr/bin/arecord",
    )
    monkeypatch.setattr(
        "inference.pi5_runner.wake_ack.subprocess.Popen",
        fake_popen,
    )

    capture = AlsaCapture(device="default:CARD=DuoCast")
    capture.start()
    capture.close()

    assert "--file-type" in commands[0]
    assert "--type" not in commands[0]


def test_beep_player_uses_supported_file_type_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        returncode = 0
        stderr = b""

    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> Result:
        commands.append(command)
        return Result()

    monkeypatch.setattr(
        "inference.pi5_runner.wake_ack.shutil.which",
        lambda _: "/usr/bin/aplay",
    )
    monkeypatch.setattr(
        "inference.pi5_runner.wake_ack.subprocess.run",
        fake_run,
    )

    AudioBeepPlayer(device="default").play()

    assert "--file-type" in commands[0]
    assert "--type" not in commands[0]


def test_beep_player_supports_pipewire_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        returncode = 0
        stderr = b""

    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> Result:
        commands.append(command)
        return Result()

    monkeypatch.setattr(
        "inference.pi5_runner.wake_ack.shutil.which",
        lambda _: "/usr/bin/pw-play",
    )
    monkeypatch.setattr(
        "inference.pi5_runner.wake_ack.subprocess.run",
        fake_run,
    )

    AudioBeepPlayer(device="pipewire:auto").play()

    assert "--target" in commands[0]
    assert commands[0][commands[0].index("--target") + 1] == "auto"
    assert commands[0][-1] == "-"


def test_speech_player_synthesizes_and_plays_with_piper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "en_US-lessac-medium.onnx"
    model_path.touch()
    Path(f"{model_path}.json").touch()
    commands: list[list[str]] = []
    voice_events: list[tuple[str, str]] = []

    class Result:
        returncode = 0
        stderr = b""

    class FakeVoice:
        config = SimpleNamespace(sample_rate=22_050)

        def synthesize_wav(self, text: str, wav_file: wave.Wave_write) -> None:
            voice_events.append(("synthesize", text))
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22_050)
            wav_file.writeframes(b"\x00\x00" * 32)

    class FakePiperVoice:
        @staticmethod
        def load(path: Path) -> FakeVoice:
            voice_events.append(("load", str(path)))
            return FakeVoice()

    def fake_run(command: list[str], **kwargs: object) -> Result:
        commands.append(command)
        with wave.open(command[-1], "rb") as wav_file:
            assert wav_file.getnchannels() == 1
        return Result()

    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(PiperVoice=FakePiperVoice),
    )
    monkeypatch.setattr(
        "inference.pi5_runner.wake_ack.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    monkeypatch.setattr(
        "inference.pi5_runner.wake_ack.subprocess.run",
        fake_run,
    )

    player = SpeechPlayer(model_path=model_path, device="pipewire:willen")
    player.speak("The local time is 6:45 PM.")
    player.speak("The local time is 6:46 PM.")

    assert len(commands) == 2
    assert all(command[1:3] == ["--target", "willen"] for command in commands)
    assert voice_events == [
        ("load", str(model_path)),
        ("synthesize", "The local time is 6:45 PM."),
        ("synthesize", "The local time is 6:46 PM."),
    ]
    assert all(not Path(command[-1]).exists() for command in commands)


def test_speech_player_warmup_synthesizes_without_playback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "voice.onnx"
    model_path.touch()
    Path(f"{model_path}.json").touch()
    synthesized: list[str] = []

    class FakeVoice:
        config = SimpleNamespace(sample_rate=22_050)

        def synthesize_wav(self, text: str, wav_file: wave.Wave_write) -> None:
            synthesized.append(text)
            wav_file.writeframes(b"\x00\x00" * 32)

    class FakePiperVoice:
        @staticmethod
        def load(path: Path) -> FakeVoice:
            return FakeVoice()

    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(PiperVoice=FakePiperVoice),
    )
    monkeypatch.setattr(
        "inference.pi5_runner.wake_ack.subprocess.run",
        lambda *args, **kwargs: pytest.fail("warmup must not play audio"),
    )

    player = SpeechPlayer(model_path=model_path, device="pipewire:willen")
    player.warmup()

    assert len(synthesized) == 1
    assert synthesized == [runtime.PIPER_WARMUP_TEXT]


def test_speech_player_reports_missing_voice_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Piper voice model"):
        SpeechPlayer(
            model_path=tmp_path / "missing.onnx",
            device="pipewire:willen",
        )


def test_speech_player_surfaces_piper_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "voice.onnx"
    model_path.touch()
    Path(f"{model_path}.json").touch()

    class FakeVoice:
        config = SimpleNamespace(sample_rate=22_050)

        def synthesize_wav(self, text: str, wav_file: wave.Wave_write) -> None:
            raise RuntimeError("synthesis failed")

    class FakePiperVoice:
        @staticmethod
        def load(path: Path) -> FakeVoice:
            return FakeVoice()

    monkeypatch.setitem(
        sys.modules,
        "piper",
        SimpleNamespace(PiperVoice=FakePiperVoice),
    )
    monkeypatch.setattr(
        "inference.pi5_runner.wake_ack.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )

    with pytest.raises(RuntimeError, match="Piper speech generation failed: synthesis failed"):
        SpeechPlayer(model_path=model_path, device="pipewire:willen").speak("Hello.")


def test_record_command_stops_after_silence_and_removes_trailing_silence() -> None:
    voiced = np.full(FRAME_SAMPLES, 4_096, dtype="<i2").tobytes()
    silence = bytes(FRAME_BYTES)
    capture = FakeCapture([voiced] * 3 + [silence] * 10)

    command = record_command(capture)

    assert command is not None
    np.testing.assert_array_equal(
        command,
        np.full(FRAME_SAMPLES * 3, 4_096, dtype=np.int16),
    )
    assert not capture.frames


def test_record_command_returns_none_when_no_speech_starts() -> None:
    capture = FakeCapture([bytes(FRAME_BYTES)] * 38)

    assert record_command(capture) is None
    assert not capture.frames


@pytest.mark.parametrize(
    ("accepted", "intent_label"),
    [
        (True, "WEATHER"),
        (False, "WEATHER"),
        (True, "CREATE_REMINDER"),
        (True, "TIMER"),
        (True, "ALARM"),
        (True, "TEMPERATURE"),
        (True, "BRIGHTNESS"),
        (True, "COLOR"),
    ],
)
def test_main_infers_and_dispatches_only_accepted_intents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    accepted: bool,
    intent_label: str,
) -> None:
    events: list[str] = []

    class Capture:
        def start(self) -> None:
            events.append("capture-start")

        def close(self) -> None:
            events.append("capture-close")

    class Detector:
        def __init__(self, model_path: Path) -> None:
            pass

        def warmup(self) -> None:
            events.append("detector-warmup")

        def reset(self) -> None:
            events.append("detector-reset")

    class Predictor:
        def __init__(self, model_path: Path, stats_path: Path) -> None:
            events.append("predictor-init")

        def predict(self, command: np.ndarray, *, threshold: float) -> object:
            assert threshold == pytest.approx(0.7)
            events.append("predict")
            return type(
                "Prediction",
                (),
                {
                    "intent_id": 1,
                    "intent_label": intent_label,
                    "confidence": 0.8,
                    "out_of_scope_confidence": 0.1,
                    "accepted": accepted,
                    "slot_ids": (1, 0, 2, 0, 1, 2),
                },
            )()

    timer_expiry_callbacks = []
    alarm_expiry_callbacks = []
    color_callbacks = []

    class Dispatcher:
        def __init__(
            self,
            *,
            timer_expiry_callback,
            alarm_expiry_callback,
            interaction_color_store,
            interaction_color_callback,
            brightness_callback,
            baresip_audio_device,
        ) -> None:
            assert interaction_color_store.load() == "Blue"
            self.timer_expiry_callback = timer_expiry_callback
            timer_expiry_callbacks.append(timer_expiry_callback)
            self.alarm_expiry_callback = alarm_expiry_callback
            alarm_expiry_callbacks.append(alarm_expiry_callback)
            color_callbacks.append(interaction_color_callback)
            assert brightness_callback is None
            assert baresip_audio_device == "test-mic"

        def execute(
            self,
            intent: str,
            *,
            slot_action: str | None = None,
        ) -> object:
            events.append(f"dispatch:{intent}:{slot_action}")
            return type(
                "Result",
                (),
                {
                    "intent": intent,
                    "status": "rejected" if not accepted else "unavailable",
                    "message": "test result",
                },
            )()

        def wait_for_call_end(self) -> None:
            events.append("wait-for-call-end")

        def close(self) -> None:
            events.append("dispatcher-close")

    args = Namespace(
        input_device="test-mic",
        indicator_hold_seconds=0.0,
        intent_model=tmp_path / "intent.onnx",
        intent_threshold=0.7,
        led_gpio=None,
        mel_stats=tmp_path / "mel_stats.npz",
        no_led=True,
        once=True,
        output_device="pipewire:auto",
        threshold=0.5,
        tts_model=tmp_path / "voice.onnx",
        wakeword_model=tmp_path / "alexa.onnx",
    )
    monkeypatch.setattr(runtime, "_parse_args", lambda argv=None: args)
    monkeypatch.setattr(runtime, "AlsaCapture", lambda **kwargs: Capture())
    monkeypatch.setattr(runtime, "OpenWakeWordDetector", Detector)
    monkeypatch.setattr(runtime, "AudioBeepPlayer", lambda **kwargs: object())
    monkeypatch.setattr(runtime, "OnnxIntentPredictor", Predictor)
    monkeypatch.setattr(runtime, "IntentDispatcher", Dispatcher)
    monkeypatch.setattr(
        runtime,
        "InteractionColorStore",
        lambda: SimpleNamespace(load=lambda: "Blue"),
    )
    monkeypatch.setattr(
        runtime,
        "SpeechPlayer",
        lambda **kwargs: SimpleNamespace(
            warmup=lambda: events.append("tts-warmup"),
            speak=lambda text: events.append(f"speak:{text}"),
        ),
    )
    monkeypatch.setattr(runtime, "listen_for_wake", lambda *a, **kw: 0.82)
    monkeypatch.setattr(
        runtime, "acknowledge_wake", lambda *a, **kw: events.append("acknowledge")
    )
    def fake_record_command(capture: Capture) -> np.ndarray:
        events.append("record-command")
        return np.zeros(10)

    monkeypatch.setattr(runtime, "record_command", fake_record_command)

    assert runtime.main() == 0

    expected_dispatch = intent_label if accepted else "OUT_OF_SCOPE"
    expected_slot_action = "None"
    if accepted and intent_label == "CREATE_REMINDER":
        expected_slot_action = "Exercise"
    elif accepted and intent_label == "TIMER":
        expected_slot_action = "30 seconds"
    elif accepted and intent_label == "ALARM":
        expected_slot_action = "6:00 AM"
    elif accepted and intent_label == "TEMPERATURE":
        expected_slot_action = "26 degrees"
    elif accepted and intent_label == "BRIGHTNESS":
        expected_slot_action = "20 percent"
    elif accepted and intent_label == "COLOR":
        expected_slot_action = "Blue"
    assert f"dispatch:{expected_dispatch}:{expected_slot_action}" in events
    assert events.index("acknowledge") < events.index("record-command")
    assert events.index("record-command") < events.index("predict")
    assert "predict" in events
    assert events.count("speak:test result") == 1
    if intent_label == "TIMER":
        timer_expiry_callbacks[0]()
        assert events.count("speak:Your timer is up.") == 1
    if intent_label == "COLOR":
        assert color_callbacks[0] is None
    if intent_label == "ALARM":
        alarm_expiry_callbacks[0]("It's 6:00 AM.")
        assert events.count("speak:It's 6:00 AM.") == 1
    assert "capture-close" in events
    assert "dispatcher-close" in events
