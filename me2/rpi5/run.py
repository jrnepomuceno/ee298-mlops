#!/usr/bin/env python3
"""Run the Pi5-VCM local replay harness."""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path

from .audio import VADConfig, microphone_utterances
from .harness import ACTION_BY_INTENT, DryRunDispatcher, HarnessConfig, PiHarness, event_json
from .replies import build_reply, reply_wav_name
from .rgb import RgbController
from .state_machine import HarnessStateMachine
from .tts import PiperTts, WavPlayer
from .timer import TimerManager
from .wakeword import OpenWakeWordDetector


LOGGER = logging.getLogger("pi5-vcm")
WILLEN_RULE = (Path.home() / ".config" / "wireplumber" / "wireplumber.conf.d"
               / "51-willen-no-idle-suspend.conf")
WILLEN_RULE_DISABLED = WILLEN_RULE.with_suffix(".conf.disabled")


def configure_logging() -> None:
    """Route runtime logs and Python warnings through one timestamped format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d\t%(levelname)s\t%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        stream=sys.stdout,
        force=True,
    )
    logging.captureWarnings(True)
    warnings.simplefilter("default")


def log_uncaught_exception(exc_type, exc_value, exc_traceback) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    LOGGER.critical("uncaught exception", exc_info=(
        exc_type, exc_value, exc_traceback))


def log(message: str, *, error: bool = False) -> None:
    """Write a timestamped human-readable runtime log line."""
    if error:
        LOGGER.error(message)
    else:
        LOGGER.info(message)


def set_demo_audio_keepalive(enabled: bool) -> bool:
    """Enable WILLEN keep-alive only for the live demo session."""
    if shutil.which("systemctl") is None:
        return False
    if enabled:
        if not WILLEN_RULE.exists() and WILLEN_RULE_DISABLED.exists():
            WILLEN_RULE_DISABLED.rename(WILLEN_RULE)
    elif WILLEN_RULE.exists():
        WILLEN_RULE.rename(WILLEN_RULE_DISABLED)
    else:
        return False
    subprocess.run(
        ["systemctl", "--user", "restart", "wireplumber"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pi5-VCM dry-run runtime harness")
    source = parser.add_argument_group("input source")
    source = source.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="16 kHz mono WAV to recognize")
    source.add_argument("--self-test", action="store_true",
                        help="run built-in synthetic utterances")
    source.add_argument("--microphone", action="store_true",
                        help="listen on the default microphone until Ctrl-C")
    source.add_argument("--vcm-only", action="store_true",
                        help="test VCM directly without a wakeword")
    model = parser.add_argument_group("model")
    model.add_argument("--checkpoint", default="inference/best.pt",
                       help="VCM checkpoint (default: inference/best.pt)")
    model.add_argument("--device", default="auto",
                       choices=["auto", "cpu", "cuda", "mps"])
    model.add_argument("--max-frames", type=int, default=400,
                       help="maximum mel frames per utterance (default: 400)")
    model.add_argument("--min-confidence", type=float, default=0.75,
                       help="minimum intent confidence (default: 0.75)")

    audio = parser.add_argument_group("audio")
    audio.add_argument("--no-ack", action="store_true",
                       help="skip wake acknowledgement playback")
    audio.add_argument("--wakeword", metavar="NAME",
                       help="pretrained wake-word model, e.g. alexa")
    audio.add_argument("--wakeword-threshold", type=float, default=0.7,
                       help="wake-word score threshold (default: 0.7)")
    audio.add_argument("--rgb-executable", metavar="PATH",
                       help="QuadcastRGB executable for DuoCast LEDs")
    audio.add_argument("--audio-player", default="pw-play",
                       help="audio player command (default: pw-play)")
    audio.add_argument("--audio-device",
                       help="PipeWire target or ALSA output device")
    audio.add_argument("--input-device",
                       help="optional PipeWire microphone target")
    audio.add_argument("--reply-dir", default="assets/replies",
                       help="static intent-reply directory")
    audio.add_argument("--ack-wav", default="assets/replies/ack_beep.wav",
                       help="wake acknowledgement WAV")

    speech = parser.add_argument_group("dynamic speech")
    speech.add_argument("--piper-bin",
                        default=str(Path.home() / "piper-venv/bin/piper"),
                        help="Piper executable for dynamic replies")
    speech.add_argument("--piper-model",
                        default=str(Path.home() /
                                    "piper-voices/en_US-lessac-medium.onnx"),
                        help="Piper voice model for dynamic replies")

    warmup = parser.add_argument_group("warmup")
    warmup.add_argument("--warmup-wav",
                        default="assets/replies/willen_mini_beep.wav",
                        help="WILLEN priming WAV")
    warmup.add_argument("--warmup-beeps", type=int, default=5,
                        help="number of priming beeps (default: 5)")
    warmup.add_argument("--warmup-beep-delay", type=float, default=4.0,
                        help="delay before priming beeps in seconds (default: 4)")
    warmup.add_argument("--no-warmup", action="store_true",
                        help="skip model and WILLEN warmup")

    diagnostics = parser.add_argument_group("diagnostics")
    diagnostics.add_argument("--wakeword-log", metavar="PATH",
                             help="append wake-word and lifecycle logs")
    diagnostics.add_argument("--wake-only", action="store_true",
                        help="test standby -> wake word -> acknowledgement only")
    diagnostics.add_argument("--dummy-time", action="store_true",
                             help="compatibility alias for --dummy-intent what_time")
    diagnostics.add_argument("--dummy-intent", choices=tuple(ACTION_BY_INTENT),
                             help="map every captured command to this intent")
    diagnostics.add_argument("--dummy-timer-minutes", type=float, default=1.0,
                             help="duration for --dummy-intent set_timer")
    diagnostics.add_argument("--warmup-color", default="FFA500",
                             help="RGB warmup color (default: FFA500)")
    return parser.parse_args()


def print_microphone_intro(args: argparse.Namespace) -> None:
    """Show the controls before entering the long-running microphone loop."""
    print("\033[2J\033[H", end="")
    log("Pi5-VCM is running")
    log("==================")
    mode = "VCM-only" if args.vcm_only else "microphone"
    log(f"Mode: {mode} / wakeword={args.wakeword or 'disabled'}")
    if args.wakeword:
        log(f"Wakeword threshold: {args.wakeword_threshold:.2f}")
        log("Say the wakeword to begin listening.")
    log("Press Ctrl-C to terminate and exit.")


def main() -> int:
    args = parse_args()
    configure_logging()
    sys.excepthook = log_uncaught_exception
    rgb = RgbController(args.rgb_executable)
    wav_player: WavPlayer | None = None
    piper_tts: PiperTts | None = None
    warmup_timer: threading.Timer | None = None
    timer_manager = TimerManager(
        on_expire=lambda state: log(
            f"[timer] expired id={state.timer_id} duration="
            f"{state.duration:g} {state.unit}"
        )
    )
    keepalive_enabled = False
    warming = not args.no_warmup
    try:
        if args.microphone or args.vcm_only:
            print_microphone_intro(args)
            wav_player = WavPlayer(args.audio_player, args.audio_device)
            if (os.path.isfile(args.piper_bin)
                    and os.path.isfile(args.piper_model)):
                piper_tts = PiperTts(args.piper_bin, args.piper_model)
            keepalive_enabled = set_demo_audio_keepalive(True)
            if keepalive_enabled:
                log("[warming] WILLEN idle-suspend disabled for demo")
        if warming:
            log("[warming] loading checkpoint and warming model")
            rgb.warming(args.warmup_color)
        harness = PiHarness(HarnessConfig(
            checkpoint=args.checkpoint,
            device=args.device,
            max_frames=args.max_frames,
            min_confidence=args.min_confidence,
            warmup=0 if args.no_warmup else 1,
        ), dispatcher=DryRunDispatcher(timer_manager))
        if warming:
            if wav_player is not None:
                warmup_path = Path(args.warmup_wav)
                if not warmup_path.exists():
                    raise FileNotFoundError(
                        f"warmup WAV not found: {warmup_path}")
                def prime_willen() -> None:
                    log(f"[warming] priming WILLEN playback with "
                        f"{args.warmup_beeps} mini-beeps")
                    for beep_number in range(args.warmup_beeps):
                        wav_player.play(str(warmup_path))
                        log(f"[warming] WILLEN mini-beep {beep_number + 1}/"
                            f"{args.warmup_beeps}")

                warmup_timer = threading.Timer(
                    args.warmup_beep_delay, prime_willen
                )
                warmup_timer.daemon = True
                warmup_timer.start()
            rgb.idle()
            log("[ready] checkpoint loaded; model warmup complete")
        if args.microphone or args.vcm_only:
            log("Valid intents: " + ", ".join(harness.intents))
        if args.input:
            events = [harness.recognize_wav(args.input)]
        elif args.self_test:
            events = harness.recognize_self_test()
        elif args.microphone or args.vcm_only:
            log_file = open(args.wakeword_log, "a", encoding="utf-8") \
                if args.wakeword_log else None

            def log_debug(message: str) -> None:
                if log_file is not None:
                    log_file.write(f"{time.time():.3f} {message}\n")
                    log_file.flush()

            log_debug("runtime_started")

            def log_score(score: float) -> None:
                if log_file is not None:
                    log_file.write(
                        f"{time.time():.3f} score={score:.6f} "
                        f"threshold={args.wakeword_threshold:.3f}\n")
                    log_file.flush()

            detector = (OpenWakeWordDetector(args.wakeword,
                                              args.wakeword_threshold,
                                              on_score=log_score)
                        if args.wakeword else None)
            vad_config = VADConfig(capture_device=args.input_device)
            current_event: dict[str, object] = {}

            if args.vcm_only:
                rgb.wake()
                log("[vcm] ready; speak now")

            def infer_command(audio):
                event = harness.recognize_audio(audio)
                forced_intent = args.dummy_intent
                if args.dummy_time:
                    forced_intent = "what_time"
                if forced_intent:
                    slots = {}
                    if forced_intent == "set_timer":
                        slots = {
                            "duration": args.dummy_timer_minutes,
                            "duration_unit": "minute",
                        }
                    result = {
                        **event["result"],
                        "intent": forced_intent,
                        "intent_confidence": 1.0,
                        "slots": slots,
                    }
                    action = harness.dispatcher.dispatch(result)
                    event["result"] = result
                    event["action"] = action
                    event["reply"] = build_reply(result, action)
                current_event.clear()
                current_event.update(event)
                return event["result"]

            def execute_action(result):
                action = harness.dispatcher.dispatch(result)
                current_event["action"] = action
                return action

            def play_reply(result, action):
                if result.get("intent") == "what_time" and piper_tts:
                    reply_text = build_reply(result, action)["text"]
                    temp = tempfile.NamedTemporaryFile(
                        prefix="pi5-vcm-time-", suffix=".wav", delete=False
                    )
                    temp.close()
                    try:
                        piper_tts.synthesize(reply_text, temp.name)
                        current_event["tts"] = wav_player.play(temp.name)
                        current_event["tts"]["source"] = "piper"
                    finally:
                        os.unlink(temp.name)
                    return

                wav_name = reply_wav_name(result, action)
                wav_path = Path(args.reply_dir) / wav_name
                if not wav_path.exists():
                    raise FileNotFoundError(f"reply WAV not found: {wav_path}")
                current_event["tts"] = wav_player.play(str(wav_path))

            def play_ack():
                if args.no_ack:
                    return None
                ack_path = str(Path(args.ack_wav))
                result = wav_player.play_async(ack_path)
                log_debug(f"ack_started wav={ack_path}")
                return result

            machine = HarnessStateMachine(
                rgb=rgb,
                play_ack=play_ack,
                capture_command=lambda _timeout: None,
                infer=infer_command,
                act=execute_action,
                play_reply=play_reply,
            )

            def acknowledge_wake() -> None:
                log("[wake] Alexa detected")
                log_debug("wake_detected")
                if detector is None:
                    return
                log("[acknowledging] acknowledgement playback skipped"
                    if args.no_ack else "[acknowledging] playing acknowledgement")
                machine.acknowledge_wake()
                log("[listening] acknowledgement complete")
                log_debug("listening_entered")

                detector.reset()

            try:
                for wav in microphone_utterances(
                    config=vad_config,
                        wakeword=detector,
                        on_wake=acknowledge_wake,
                        on_speech=rgb.wake,
                        on_timeout=machine.cancel_listening,
                        command_timeout_s=machine.command_timeout_s,
                        wake_only=args.wake_only):
                    if args.wake_only:
                        continue
                    outcome = machine.handle_command(wav)
                    if outcome is not None:
                        result = current_event.get("result")
                        intent = (result.get("intent", "unknown")
                                  if isinstance(result, dict) else "unknown")
                        log(f"[vcm] intent: {intent}")
                        current_event["rgb"] = rgb.off()
                        print(event_json(current_event), flush=True)
                        if args.vcm_only:
                            rgb.wake()
                            log("[vcm] ready; speak now")
            finally:
                if log_file is not None:
                    log_file.close()
            events = []
    except KeyboardInterrupt:
        log("[shutdown] Pi5-VCM stopped.")
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log(f"error: {exc}", error=True)
        return 2
    finally:
        if warmup_timer is not None:
            warmup_timer.cancel()
            if warmup_timer.is_alive():
                warmup_timer.join(timeout=2)
        if keepalive_enabled:
            try:
                set_demo_audio_keepalive(False)
                log("[shutdown] WILLEN idle-suspend restored")
            except (OSError, subprocess.SubprocessError) as exc:
                log(f"[shutdown] failed to restore WILLEN idle-suspend: {exc}",
                    error=True)
        if wav_player is not None:
            wav_player.close()
        timer_manager.close()
        rgb.close()

    for event in events:
        print(event_json(event))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
