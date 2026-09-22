#!/usr/bin/env python3
"""Run the Pi5-VCM local replay harness."""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

from .audio import VADConfig, microphone_utterances
from .harness import HarnessConfig, PiHarness, event_json
from .replies import reply_wav_name
from .rgb import RgbController
from .state_machine import HarnessStateMachine
from .tts import WavPlayer
from .wakeword import OpenWakeWordDetector


LOGGER = logging.getLogger("pi5-vcm")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pi5-VCM dry-run runtime harness")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="16 kHz mono WAV to recognize")
    source.add_argument("--self-test", action="store_true",
                        help="run built-in synthetic utterances")
    source.add_argument("--microphone", action="store_true",
                        help="listen on the default microphone until Ctrl-C")
    source.add_argument("--vcm-only", action="store_true",
                        help="test VCM directly without a wakeword")
    parser.add_argument("--checkpoint", default="inference/best.pt")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--no-ack", action="store_true",
                        help="skip wake acknowledgement WAV playback")
    parser.add_argument("--wakeword", metavar="NAME",
                        help="optional pretrained openWakeWord model, e.g. alexa")
    parser.add_argument("--wakeword-threshold", type=float, default=0.7)
    parser.add_argument("--wakeword-log", metavar="PATH",
                        help="append wake-word scores to this log")
    parser.add_argument("--rgb-executable", metavar="PATH",
                        help="optional quadcastrgb executable for DuoCast LEDs")
    parser.add_argument("--audio-player", default="pw-play",
                        help="PCM WAV player (default: pw-play)")
    parser.add_argument("--audio-device",
                        help="ALSA device passed to aplay, e.g. plughw:2,0")
    parser.add_argument("--input-device",
                        help="microphone device name; default is system input")
    parser.add_argument("--reply-dir", default="assets/replies",
                        help="directory containing static intent reply WAVs")
    parser.add_argument("--ack-wav", default="assets/replies/yes.wav",
                        help="static wake acknowledgement WAV")
    parser.add_argument("--wake-only", action="store_true",
                        help="test standby -> wake word -> acknowledgement only")
    parser.add_argument("--no-warmup", action="store_true")
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
    try:
        if args.microphone or args.vcm_only:
            print_microphone_intro(args)
        harness = PiHarness(HarnessConfig(
            checkpoint=args.checkpoint,
            device=args.device,
            max_frames=args.max_frames,
            min_confidence=args.min_confidence,
            warmup=0 if args.no_warmup else 1,
        ))
        if args.microphone or args.vcm_only:
            log("Valid intents: " + ", ".join(harness.intents))
        if args.input:
            events = [harness.recognize_wav(args.input)]
        elif args.self_test:
            events = harness.recognize_self_test()
        elif args.microphone or args.vcm_only:
            log_file = open(args.wakeword_log, "a", encoding="utf-8") \
                if args.wakeword_log else None

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
            rgb = RgbController(args.rgb_executable)
            wav_player = WavPlayer(args.audio_player, args.audio_device)
            vad_config = VADConfig(capture_device=args.input_device)
            current_event: dict[str, object] = {}

            if args.vcm_only:
                rgb.wake()
                log("[vcm] ready; speak now")

            def infer_command(audio):
                event = harness.recognize_audio(audio)
                current_event.clear()
                current_event.update(event)
                return event["result"]

            def execute_action(result):
                action = harness.dispatcher.dispatch(result)
                current_event["action"] = action
                return action

            def play_reply(result, action):
                wav_name = reply_wav_name(result, action)
                wav_path = Path(args.reply_dir) / wav_name
                if not wav_path.exists():
                    raise FileNotFoundError(f"reply WAV not found: {wav_path}")
                current_event["tts"] = wav_player.play(str(wav_path))

            machine = HarnessStateMachine(
                rgb=rgb,
                play_ack=(lambda: None if args.no_ack else
                          wav_player.play(str(Path(args.ack_wav)))),
                capture_command=lambda _timeout: None,
                infer=infer_command,
                act=execute_action,
                play_reply=play_reply,
            )

            def acknowledge_wake() -> None:
                log("[wake] Alexa detected")
                if detector is None:
                    return
                log("[acknowledging] yes.wav playback skipped"
                    if args.no_ack else "[acknowledging] playing yes.wav")
                machine.acknowledge_wake()
                log("[listening] acknowledgement complete")

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
                rgb.close()
                if log_file is not None:
                    log_file.close()
            events = []
    except KeyboardInterrupt:
        log("[shutdown] Pi5-VCM stopped.")
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        log(f"error: {exc}", error=True)
        return 2

    for event in events:
        print(event_json(event))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
