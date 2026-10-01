"""Run all intent scenarios through the real facade/orchestrator without ONNX or hardware."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

from .audio import VADConfig, microphone_utterances
from .facade import INTENT_SPECS
from .harness import FacadePipeline
from .mock_inference import MOCK_SLOTS, MockInference
from .orchestrator import default_orchestrator
from .calls import make_dialer, normalize_target
from .rgb import RgbController, make_light_driver
from .tts import PiperTts, WavPlayer
from .timer import TimerAlarm, TimerManager, TimerState
from .volume import VolumeController
from .weather import make_weather_fn


class _RecordingRgb:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def processing(self) -> None:
        self.calls.append("processing")

    def speaking(self) -> None:
        self.calls.append("speaking")

    def idle(self) -> None:
        self.calls.append("idle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent", choices=("all", *MOCK_SLOTS), default="all")
    parser.add_argument("--microphone", action="store_true",
                        help="capture one real utterance before running the selected mock intent")
    parser.add_argument("--input-device", default=None,
                        help="PipeWire/sounddevice capture device (default: system default)")
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--temperature", type=int, default=22,
                        help="mock temperature set-point in Celsius (10-35)")
    parser.add_argument("--live-speaker", action="store_true",
                        help="play the reply through the real speaker")
    parser.add_argument("--speaker-wav", default=None,
                        help="play this existing WAV instead of synthesizing with Piper")
    parser.add_argument("--piper-bin", default=str(Path.home() / "piper-venv/bin/piper"))
    parser.add_argument("--piper-model",
                        default=str(Path.home() / "piper-voices/en_US-lessac-medium.onnx"))
    parser.add_argument("--audio-player", default="aplay")
    parser.add_argument("--audio-device", default=None)
    parser.add_argument("--live-lights", action="store_true",
                        help="allow the selected light intent to drive a real light")
    parser.add_argument("--live-volume", action="store_true",
                        help="adjust the real master output for one volume intent, then restore it")
    parser.add_argument("--live-timer", action="store_true",
                        help="start/cancel a real process-local timer for a timer intent")
    parser.add_argument("--live-rgb", action="store_true",
                        help="show the timer alarm cue on a real RGB controller")
    parser.add_argument("--rgb-executable", default=None,
                        help="QuadcastRGB executable used by --live-rgb")
    parser.add_argument("--timer-seconds", type=int, default=3,
                        help="short test timer duration in seconds (1-30)")
    parser.add_argument("--ring-seconds", type=int, default=3,
                        help="keep the test alarm ringing before injecting mock stop_timer (1-30)")
    parser.add_argument("--timer-expiry-wav", default="assets/replies/timer_ding.wav")
    parser.add_argument("--light-driver", default="hyperx-duocast")
    parser.add_argument("--light-executable", default=None)
    parser.add_argument("--live-weather", action="store_true",
                        help="make a real OpenWeatherMap request for what_weather")
    parser.add_argument("--weather-location", default=None)
    parser.add_argument("--live-call", action="store_true",
                        help="place a real SIP call; requires --confirm-call and --mock-contact")
    parser.add_argument("--confirm-call", action="store_true",
                        help="explicitly authorize the single mocked SIP call")
    parser.add_argument("--mock-contact", default=None,
                        help="numeric SIP target used by the call scenario")
    parser.add_argument("--call-seconds", type=int, default=15,
                        help="maximum duration for the explicitly confirmed SIP test call")
    parser.add_argument("--baresip-bin", default=None)
    parser.add_argument("--baresip-config", default=None)
    parser.add_argument("--baresip-account", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0 and 1")
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0 and 1")
    if not 10 <= args.temperature <= 35:
        raise SystemExit("--temperature must be between 10 and 35 Celsius")
    if args.microphone and args.intent == "all":
        raise SystemExit("--microphone requires one explicit --intent")

    live_flags = (args.live_speaker, args.live_lights, args.live_volume,
                  args.live_timer, args.live_rgb,
                  args.live_weather, args.live_call)
    if any(live_flags) and args.intent == "all":
        raise SystemExit("live peripheral tests require one explicit --intent")
    if args.live_lights and args.intent not in {
            "turn_on_lights", "turn_off_lights", "dim_lights"}:
        raise SystemExit("--live-lights requires a lights intent")
    if args.live_volume and args.intent not in {"volume_up", "volume_down"}:
        raise SystemExit("--live-volume requires --intent volume_up or volume_down")
    if args.live_volume and args.microphone:
        raise SystemExit("--live-volume runs one mock command without microphone capture")
    if args.live_timer:
        if args.intent not in {"set_timer", "stop_timer"}:
            raise SystemExit("--live-timer requires --intent set_timer or stop_timer")
        if not args.live_speaker:
            raise SystemExit("--live-timer requires --live-speaker for audible confirmation")
        if not 1 <= args.timer_seconds <= 30:
            raise SystemExit("--timer-seconds must be between 1 and 30")
        if not 1 <= args.ring_seconds <= 30:
            raise SystemExit("--ring-seconds must be between 1 and 30")
    if args.live_rgb:
        if not args.live_timer:
            raise SystemExit("--live-rgb requires --live-timer")
        if args.intent not in {"set_timer", "stop_timer"}:
            raise SystemExit("--live-rgb requires a timer intent")
        if not args.rgb_executable or not (
                Path(args.rgb_executable).is_file() or
                shutil.which(args.rgb_executable)):
            raise SystemExit("--live-rgb requires an available --rgb-executable")
    if args.live_weather and args.intent != "what_weather":
        raise SystemExit("--live-weather requires --intent what_weather")
    if args.live_call:
        if args.intent != "call":
            raise SystemExit("--live-call requires --intent call")
        if not args.confirm_call:
            raise SystemExit("--live-call requires --confirm-call")
        if not args.mock_contact:
            raise SystemExit("--live-call requires --mock-contact with a numeric target")
        try:
            args.mock_contact = normalize_target(args.mock_contact)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if not 1 <= args.call_seconds <= 120:
            raise SystemExit("--call-seconds must be between 1 and 120")
    elif args.confirm_call or args.mock_contact:
        raise SystemExit("--confirm-call and --mock-contact are only valid with --live-call")

    piper = player = None
    if args.live_speaker:
        if args.speaker_wav and not Path(args.speaker_wav).is_file():
            raise SystemExit(f"speaker WAV not found: {args.speaker_wav}")
        if not args.speaker_wav and (
                not Path(args.piper_bin).is_file() or
                not Path(args.piper_model).is_file()):
            raise SystemExit("--live-speaker requires an existing --piper-bin and --piper-model")
        if not (Path(args.audio_player).is_file() or shutil.which(args.audio_player)):
            raise SystemExit(f"audio player not found: {args.audio_player}")
        if not args.speaker_wav:
            piper = PiperTts(args.piper_bin, args.piper_model)
        player = WavPlayer(args.audio_player, args.audio_device)
    elif args.speaker_wav:
        raise SystemExit("--speaker-wav requires --live-speaker")
    if args.live_timer:
        if piper is None or player is None:
            raise SystemExit("--live-timer requires Piper and an audio player")
        if not Path(args.timer_expiry_wav).is_file():
            raise SystemExit(f"timer expiry WAV not found: {args.timer_expiry_wav}")

    light_driver = None
    if args.live_lights:
        light_driver = make_light_driver(args.light_driver, args.light_executable)
        if light_driver is None or not light_driver.available():
            raise SystemExit("selected light driver is unavailable")

    rgb_controller = (RgbController(args.rgb_executable, light_driver=light_driver)
                      if args.live_rgb else None)
    timer_rgb_report = ({"alert_started": None, "alert_stopped": None}
                        if rgb_controller is not None else None)

    dialer = None
    if args.live_call:
        dialer = make_dialer("baresip", executable=args.baresip_bin,
                             config=args.baresip_config,
                             account=args.baresip_account)
        if dialer is None or not dialer.available():
            raise SystemExit("Baresip is unavailable; refusing to attempt a live call")

    volume_controller = None
    volume_before = None
    if args.live_volume:
        volume_controller = VolumeController()
        volume_before = volume_controller.begin()
        if volume_before is None:
            raise SystemExit("could not read current output volume; refusing to change it")

    rgb = _RecordingRgb()
    spoken: list[str] = []
    events: list[dict] = []
    playback: list[dict] = []
    microphone_capture = None
    volume_report = ({"before_percent": volume_before,
                      "after_action_percent": None,
                      "restore_ok": None,
                      "after_restore_percent": None}
                     if volume_controller is not None else None)

    if args.microphone:
        print("[mock] Speak one command now; its audio triggers capture only, "
              "the selected intent is forced.", file=sys.stderr, flush=True)
        utterances = microphone_utterances(
            config=VADConfig(capture_device=args.input_device),
            wakeword=None,
            on_speech=lambda: print("[mock] speech detected", file=sys.stderr,
                                    flush=True),
        )
        try:
            waveform = next(utterances)
        except StopIteration as exc:
            raise SystemExit("microphone stream ended before an utterance") from exc
        finally:
            utterances.close()
        microphone_capture = {
            "samples": len(waveform),
            "duration_seconds": round(len(waveform) / VADConfig().sample_rate, 3),
            "input_device": args.input_device or "system-default",
        }

    def play_wav(wav_path: str) -> None:
        try:
            playback.append(player.play(wav_path))
        except Exception as exc:  # noqa: BLE001 - surface live-device failures in the report
            playback.append({"status": "error", "wav": wav_path,
                             "error": str(exc)})
            raise

    def speak(text: str) -> None:
        spoken.append(text)
        if player is None:
            return
        if args.speaker_wav:
            play_wav(args.speaker_wav)
            return
        if piper is None:
            return
        with tempfile.NamedTemporaryFile(prefix="vcm-mock-", suffix=".wav",
                                         delete=False) as temp:
            wav_path = temp.name
        try:
            piper.synthesize(text, wav_path)
            play_wav(wav_path)
        finally:
            os.unlink(wav_path)

    timer_expired = threading.Event()
    timer_report = None
    timer_manager = None
    timer_alarm = None
    if args.live_timer:
        timer_report = {
            "test_duration_seconds": args.timer_seconds,
            "ring_observation_seconds": args.ring_seconds,
            "ring_play_count": 0,
            "expiry_announcement_count": 0,
            "expiry_error": None,
        }

        def play_timer_ring() -> None:
            timer_report["ring_play_count"] += 1
            try:
                play_wav(args.timer_expiry_wav)
            except Exception as exc:  # noqa: BLE001 - report async ring failures
                timer_report["expiry_error"] = str(exc)
                raise

        def announce_timer_expiry() -> None:
            timer_report["expiry_announcement_count"] += 1
            try:
                speak("Your timer is up.")
            except Exception as exc:  # noqa: BLE001 - report async speech failures
                timer_report["expiry_error"] = str(exc)
                raise

        def start_timer_rgb_alert() -> None:
            timer_rgb_report["alert_started"] = rgb_controller.timer_alert()

        def stop_timer_rgb_alert() -> None:
            timer_rgb_report["alert_stopped"] = rgb_controller.clear_timer_alert()

        timer_alarm = TimerAlarm(
            play_timer_ring,
            announce_timer_expiry,
            announce_interval_s=10.0,
            ring_gap_s=1.0,
            on_started=(start_timer_rgb_alert if rgb_controller is not None else None),
            on_stopped=(stop_timer_rgb_alert if rgb_controller is not None else None),
        )

        def on_timer_expire(state: TimerState) -> None:
            timer_report["expiry_callback_received"] = True
            timer_report["alarm_started"] = timer_alarm.start()
            timer_expired.set()

        timer_manager = TimerManager(on_expire=on_timer_expire)

    timer_before_start = None
    if args.live_timer and player is not None:
        def announce_timer_start(text: str) -> bool:
            try:
                speak(text)
                return True
            except Exception as exc:  # noqa: BLE001 - keep timer setup fail-soft
                print(f"[mock] timer announcement failed: {exc}",
                      file=sys.stderr, flush=True)
                return False

        timer_before_start = announce_timer_start

    orchestrator = default_orchestrator(
        dry_run=not (args.live_lights or args.live_volume or args.live_timer or
                     args.live_weather or args.live_call),
        rgb=rgb_controller or rgb,
        speak=speak,
        on_event=events.append,
        light_driver=light_driver,
        volume_controller=volume_controller,
        timer_manager=timer_manager,
        timer_alarm=timer_alarm,
        timer_before_start=timer_before_start,
        weather_fn=(make_weather_fn(args.weather_location)
                    if args.live_weather else None),
        dialer=dialer,
    )
    pipeline = FacadePipeline(orchestrator=orchestrator,
                              threshold=args.threshold,
                              dry_run=True)
    inference = MockInference()
    intents = tuple(INTENT_SPECS) + ("oov",) if args.intent == "all" else (args.intent,)

    try:
        for intent in intents:
            overrides = ({"temperature": args.temperature}
                         if intent == "set_temperature" else None)
            if args.live_timer and intent == "set_timer":
                overrides = {"duration": args.timer_seconds,
                             "duration_unit": "second"}
            if args.live_timer and intent == "stop_timer":
                timer_manager.set_timer(args.timer_seconds, "second")
                timer_report["prearmed_for_cancel"] = True
            result = inference.recognize(
                intent, args.confidence, slot_overrides=overrides)
            if intent == "call" and args.live_call:
                result["slots"]["contact"] = args.mock_contact
            outcome = pipeline.process(
                result, source="microphone" if args.microphone else "mock")
            if args.live_timer:
                timer_report["action"] = outcome.execution.action_code
                timer_report["action_side_effects"] = outcome.execution.side_effects
                if intent == "set_timer":
                    print(f"[mock] timer started for {args.timer_seconds}s; "
                          "waiting for expiry then a persistent looping alarm",
                          file=sys.stderr, flush=True)
                    timer_report["expiry_event_received"] = timer_expired.wait(
                        timeout=args.timer_seconds + 5)
                    if not timer_report["expiry_event_received"]:
                        timer_report["expiry_error"] = "timer expiry callback timed out"
                    else:
                        timer_report["ring_started"] = timer_alarm.wait_first_ring(timeout=5)
                        if not timer_report["ring_started"]:
                            timer_report["expiry_error"] = "alarm loop did not start"
                        else:
                            print(f"[mock] alarm is looping; injecting mock stop_timer "
                                  f"after {args.ring_seconds}s", file=sys.stderr,
                                  flush=True)
                            threading.Event().wait(args.ring_seconds)
                            stop_result = inference.recognize("stop_timer",
                                                              args.confidence)
                            stop_outcome = pipeline.process(stop_result, source="mock")
                            timer_report["stop_intent_received"] = True
                            timer_report["stop_action_side_effects"] = (
                                stop_outcome.execution.side_effects
                                if stop_outcome.execution else False)
                            timer_report["alarm_stopped"] = not timer_alarm.active
                else:
                    print(f"[mock] timer prearmed; checking cancellation for "
                          f"{args.timer_seconds + 1}s", file=sys.stderr, flush=True)
                    timer_report["expiry_event_received"] = timer_expired.wait(
                        timeout=args.timer_seconds + 1)
                    timer_report["cancel_prevented_expiry"] = not timer_report[
                        "expiry_event_received"]
            if volume_controller is not None:
                volume_report["after_action_percent"] = volume_controller.get()
            if args.live_call:
                print(f"[mock] holding confirmed SIP test call for {args.call_seconds}s",
                      file=sys.stderr, flush=True)
                time.sleep(args.call_seconds)

        if volume_controller is not None:
            volume_report["restore_ok"] = volume_controller.end()
            volume_report["after_restore_percent"] = volume_controller.get()

        print(json.dumps({
            "backend": "mock",
            "live_components": {
                "microphone": args.microphone,
                "speaker": args.live_speaker,
                "lights": args.live_lights,
                "timer_rgb": args.live_rgb,
                "weather_api": args.live_weather,
                "sip_call": args.live_call,
                "timer": args.live_timer,
            },
            "scenario_count": len(events),
            "mock_intents": [event.get("intent") for event in events],
            "microphone_capture": microphone_capture,
            "timer": timer_report,
            "timer_rgb": timer_rgb_report,
            "volume": volume_report,
            "events": events,
            "spoken_replies": spoken,
            "playback": playback,
            "rgb_calls": rgb.calls,
        }, indent=2, default=str))
    finally:
        if light_driver is not None:
            if args.intent not in {"turn_on_lights", "dim_lights"}:
                light_driver.close()
        if dialer is not None:
            dialer.close()
        if player is not None:
            player.close()
        if volume_controller is not None:
            volume_controller.end()
        if timer_manager is not None:
            timer_manager.close()
        if timer_alarm is not None:
            timer_alarm.stop()
        if rgb_controller is not None:
            rgb_controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
