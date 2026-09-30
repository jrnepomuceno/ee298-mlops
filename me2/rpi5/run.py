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
from .harness import ACTION_BY_INTENT, DryRunDispatcher, FacadePipeline, HarnessConfig, PiHarness, event_json
from .replies import build_reply, reply_wav_name
from .rgb import RgbController, make_light_driver
from .calls import make_dialer
from .state_machine import HarnessStateMachine
from .tts import PiperTts, WavPlayer
from .reminders import ReminderStore
from .timer import AlarmState, TimerAlarm, TimerManager, TimerState
from .weather import make_weather_fn
from .volume import VolumeController
from .media import MediaPlayerController
from .wakeword import OpenWakeWordDetector

try:  # torch-free ONNX ground-truth resolver (lives at project root)
    from model.onnx_deploy import DEFAULT_ONNX_DIR, resolve_latest_onnx
except ImportError:  # pragma: no cover - run.py executed standalone
    DEFAULT_ONNX_DIR = None
    resolve_latest_onnx = None


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
    _gt = (resolve_latest_onnx(DEFAULT_ONNX_DIR, "int8")
           if resolve_latest_onnx is not None else None)
    _ckpt_default = str(_gt) if _gt else "vcm_model_int8.onnx"
    model.add_argument("--checkpoint", default=_ckpt_default,
                       help="VCM ONNX model (default: newest models/onnx/<tag>/vcm_model_int8.onnx)")
    model.add_argument("--intent-labels", default=None,
                       help="enable diagnostics-only mode for a single-output intent ONNX and load its label JSON")
    model.add_argument("--device", default="cpu",
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="informational; the ONNX path always runs on CPU")
    model.add_argument("--threads", type=int, default=2,
                       help="ORT intra_op_num_threads (default: 2 for Pi5)")
    model.add_argument("--max-frames", type=int, default=400,
                       help="maximum mel frames per utterance (default: 400)")
    model.add_argument("--min-confidence", type=float, default=0.75,
                       help="minimum intent confidence (default: 0.75)")

    audio = parser.add_argument_group("audio")
    audio.add_argument("--no-ack", action="store_true",
                       help="skip wake acknowledgement playback")
    audio.add_argument("--post-reply-guard", type=float, default=4.0,
                       metavar="SECONDS",
                       help="keep the microphone closed this long after a "
                            "reply finishes, so the trailing room echo cannot "
                            "re-latch the wake detector (default: 4.0)")
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
    audio.add_argument("--timer-expiry-wav", default="assets/replies/timer_ding.wav",
                       help="WAV played when a countdown timer expires")
    audio.add_argument("--alarm-expiry-wav", default="assets/replies/alarm_ring.wav",
                       help="WAV played when a wall-clock alarm rings")

    speech = parser.add_argument_group("dynamic speech")
    speech.add_argument("--piper-bin",
                        default=str(Path.home() / "piper-venv/bin/piper"),
                        help="Piper executable for dynamic replies")
    speech.add_argument("--piper-model",
                        default=str(Path.home() /
                                    "piper-voices/en_US-lessac-medium.onnx"),
                        help="Piper voice model for dynamic replies")

    weather = parser.add_argument_group("weather")
    weather.add_argument("--weather-location", default=None,
                         help="City for what_weather (default: $WEATHER_LOCATION or Quezon City)")
    weather.add_argument("--no-weather", action="store_true",
                         help="Disable live weather lookup (use the offline fallback line)")

    reminders = parser.add_argument_group("reminders")
    reminders.add_argument("--reminders-file", default=None,
                           help="Path to the reminders JSON file "
                                "(default: ~/.me2/reminders.json)")

    volume = parser.add_argument_group("volume")
    volume.add_argument("--no-volume", action="store_true",
                        help="Disable live output-volume control (volume_up/"
                            "volume_down become dry-run only)")

    music = parser.add_argument_group("music")
    music.add_argument("--music-dir", default=str(Path.home() / "Music"),
                       help="Directory of audio files for play_music "
                            "(default: ~/Music)")
    music.add_argument("--live-media", action="store_true",
                       help="Enable live music playback (play_music/pause_"
                            "music/stop_music). Off by default: media intents "
                            "stay dry-run.")
    music.add_argument("--player", default=None,
                       help="Force a specific audio player binary "
                            "(default: auto-detect ffplay/pw-play/aplay)")
    music.add_argument("--media-volume", type=int, default=None,
                       metavar="PCT",
                       help="Per-app PipeWire volume for the music stream "
                            "(0-100). Independent of the master sink that "
                            "TTS/alarms use. Requires --live-media and a "
                            "wpctl-capable Pi; ignored elsewhere.")

    lights = parser.add_argument_group("lights")
    lights.add_argument("--light-driver", default="hyperx-duocast",
                        metavar="NAME",
                        help="Light device behind the voice commands "
                             "(turn_on_lights/turn_off_lights/dim_lights). "
                             "Default: hyperx-duocast (the HyperX DuoCast "
                             "mic ring light). Pass an empty string to keep "
                             "lights dry-run.")
    lights.add_argument("--light-executable", metavar="PATH", default=None,
                        help="Override the driver's control executable "
                             "(default: $QUADCASTRGB or 'quadcastrgb' on PATH)")

    calls = parser.add_argument_group("calls")
    calls.add_argument("--dialer", default="baresip",
                       metavar="NAME",
                       help="Dial device behind the call intent. "
                            "Default: baresip (SIP softphone). "
                            "Pass an empty string to keep calls dry-run.")
    calls.add_argument("--baresip-config", metavar="PATH", default=None,
                       help="baresip config/profile file (accounts, transports, "
                            "audio). Passed as '--config' to the daemon. "
                            "Default: baresip's own default profile.")
    calls.add_argument("--baresip-account", metavar="ADDR", default=None,
                       help="Pin which registered baresip account dials "
                            "(used when the profile has several accounts). "
                            "Passed as '--account'.")
    calls.add_argument("--baresip-bin", metavar="PATH", default=None,
                       help="Override the baresip executable "
                            "(default: $BARESIP or 'baresip' on PATH)")

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
    if args.intent_labels and not (args.input or args.self_test):
        raise SystemExit("--intent-labels is diagnostics-only; use --input or --self-test (no microphone/actions)")
    configure_logging()
    sys.excepthook = log_uncaught_exception
    rgb = RgbController(args.rgb_executable)
    wav_player: WavPlayer | None = None
    piper_tts: PiperTts | None = None
    tts_active = threading.Event()
    media_before_play = None
    timer_alarm: TimerAlarm | None = None
    warmup_timer: threading.Timer | None = None
    def _on_timer_expire(state) -> None:
        # Fires from the timer thread, not the mic loop: keep it self-contained
        # and fail-soft so a missing speaker/TTS never takes the process down.
        if isinstance(state, TimerState):
            if timer_alarm is None:
                log("[timer] expired but no audio alarm is available", error=True)
            elif timer_alarm.start():
                log(f"[timer] expired id={state.timer_id}; repeating alarm started")
            return
        if not isinstance(state, AlarmState):
            return

        def _expiry_play(play_fn, path, kind):
            # Duck the master volume under the ring/speech so the mic can still
            # hear the wake word (Echo-style), then restore. Self-contained so
            # the timer thread never depends on the mic-loop closures.
            # Ducking is disabled: the master dip caused audible volume
            # pumping and still leaked into the mic. Play at full volume.
            ctrl = volume_controller
            if False:  # ducking disabled
                try:
                    ctrl.duck()
                except Exception:  # noqa: BLE001
                    log(f"[{kind}] duck failed; playing at full volume",
                        error=True)
            tts_active.set()
            try:
                play_fn(path)
            finally:
                if False:  # ducking disabled
                    try:
                        ctrl.unduck()
                    except Exception:  # noqa: BLE001
                        log(f"[{kind}] unduck failed", error=True)
                tts_active.clear()

            kind = "alarm"
            log(f"[alarm] ringing id={state.alarm_id} time={state.time}")
            wav_path = args.alarm_expiry_wav
            speech_text = f"It's {state.time}."
        try:
            if wav_player is not None:
                _expiry_play(wav_player.play, wav_path, kind)
        except Exception:  # noqa: BLE001
            log(f"[{kind}] expiry beep failed", exc_info=True)
        try:
            if piper_tts is not None:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                piper_tts.synthesize(speech_text, tmp_path)
                _expiry_play(wav_player.play, tmp_path, kind)
        except Exception:  # noqa: BLE001
            log(f"[{kind}] expiry speech failed", exc_info=True)

    timer_manager = TimerManager(on_expire=_on_timer_expire)
    volume_controller = (None if args.no_volume
                          else VolumeController())
    media_player = (MediaPlayerController(args.music_dir, player=args.player)
                     if args.live_media else None)
    light_driver = make_light_driver(args.light_driver, args.light_executable)
    if light_driver is not None:
        if light_driver.available():
            log(f"[lights] live driver '{light_driver.name}' ready")
        else:
            log(f"[lights] driver '{light_driver.name}' selected but its "
                "executable is not on this host; light commands will report "
                "unreachable instead of driving hardware")
    else:
        log("[lights] no driver selected; light intents stay dry-run")

    dialer = make_dialer(args.dialer, executable=args.baresip_bin,
                         config=args.baresip_config,
                         account=args.baresip_account)
    if dialer is not None:
        if dialer.available():
            log(f"[calls] live dialer '{dialer.name}' ready")
        else:
            log(f"[calls] dialer '{dialer.name}' selected but its "
                "executable is not on this host; call commands will report "
                "unreachable instead of placing calls")
    else:
        log("[calls] no dialer selected; call intent stays dry-run")
    keepalive_enabled = False
    warming = not args.no_warmup
    try:
        if args.microphone or args.vcm_only:
            print_microphone_intro(args)
            wav_player = WavPlayer(args.audio_player, args.audio_device)
            if (os.path.isfile(args.piper_bin)
                    and os.path.isfile(args.piper_model)):
                piper_tts = PiperTts(args.piper_bin, args.piper_model)

            def announce_music_track(track_name: str) -> bool:
                nonlocal post_reply_until
                if piper_tts is None or wav_player is None:
                    return False
                reply_text = f"Playing {Path(track_name).stem}"
                with tempfile.NamedTemporaryFile(
                        prefix="pi5-vcm-music-", suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                tts_active.set()
                try:
                    piper_tts.synthesize(reply_text, tmp_path)
                    wav_player.play(tmp_path)
                    time.sleep(0.3)
                    if post_reply_guard_s > 0:
                        post_reply_until = time.monotonic() + post_reply_guard_s
                    return True
                except Exception as exc:  # noqa: BLE001 - keep live music fail-soft
                    log(f"[media] pre-play announcement failed: {exc}", error=True)
                    return False
                finally:
                    tts_active.clear()
                    os.unlink(tmp_path)

            media_before_play = announce_music_track

            def play_timer_ring() -> None:
                if wav_player is None:
                    return
                tts_active.set()
                try:
                    wav_player.play(args.timer_expiry_wav)
                finally:
                    tts_active.clear()

            def announce_timer_expiry() -> None:
                if piper_tts is None or wav_player is None:
                    log("[timer] Piper unavailable; skipping spoken expiry reminder",
                        error=True)
                    return
                with tempfile.NamedTemporaryFile(
                        prefix="pi5-vcm-timer-", suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                tts_active.set()
                try:
                    piper_tts.synthesize("Your timer is up.", tmp_path)
                    wav_player.play(tmp_path)
                finally:
                    tts_active.clear()
                    os.unlink(tmp_path)

            timer_alarm = TimerAlarm(
                play_timer_ring,
                announce_timer_expiry,
                announce_interval_s=10.0,
                ring_gap_s=1.0,
                on_started=rgb.timer_alert,
                on_stopped=rgb.clear_timer_alert,
            )
            keepalive_enabled = set_demo_audio_keepalive(True)
            if keepalive_enabled:
                log("[warming] WILLEN idle-suspend disabled for demo")
        if warming:
            log("[warming] loading checkpoint and warming model")
            rgb.warming(args.warmup_color)
        weather_fn = None if args.no_weather else make_weather_fn(args.weather_location)
        reminder_store = ReminderStore(args.reminders_file)
        if volume_controller is not None:
            volume_controller.begin()  # snapshot pre-demo level for restore
        pipeline = FacadePipeline(threshold=args.min_confidence,
                                  # Live mode: persist reminders and schedule
                                  # timers for real. (Was dry_run=True, which
                                  # made remind/timer confirm but never act.)
                                  dry_run=False,
                                  weather_fn=weather_fn,
                                  timer_manager=timer_manager,
                                  timer_alarm=timer_alarm,
                                  reminder_store=reminder_store,
                                  volume_controller=volume_controller,
                                  media_player=media_player,
                                  media_volume=args.media_volume,
                                  media_before_play=media_before_play,
                                  light_driver=light_driver,
                                  dialer=dialer)
        harness = PiHarness(HarnessConfig(
            checkpoint=args.checkpoint,
            intent_labels_path=args.intent_labels,
            device=args.device,
            max_frames=args.max_frames,
            min_confidence=args.min_confidence,
            warmup=0 if args.no_warmup else 1,
            threads=args.threads,
        ), dispatcher=DryRunDispatcher(timer_manager), pipeline=pipeline)
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
            # Shared "assistant is producing audio" flag. Raised by
            # _ducked_play for the full playback window (including the blocking
            # play() call) and read by the microphone loop, which closes the mic
            # while it is set. This is the echo guard that stops a spoken reply
            # from being recognised as a second command.
            # Monotonic deadline until which the microphone stays closed after
            # a reply finishes. The 300 ms tail inside _ducked_play absorbs
            # speaker ring-down, but the *room* keeps echoing the reply for a
            # couple of seconds; that lingering echo is what re-latches the
            # wake detector and produces the phantom "Alexa detected" + OOV
            # reply. Holding the mic closed for post_reply_guard_s after every
            # reply (including the OOV reply itself) kills that loop.
            post_reply_guard_s = max(0.0, args.post_reply_guard)
            post_reply_until = 0.0
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
                else:
                    result = event["result"]
                # Recognize only. Do NOT execute here: the state machine's
                # handle_command() -> act() -> execute_action() is the single
                # point where the action runs and the reply is spoken. Executing
                # in infer_command as well made every command fire twice
                # ("two Alexas speaking").
                event["result"] = result
                current_event.clear()
                current_event.update(event)
                return result

            def execute_action(result):
                # Single execution path. The facade runs the action once and
                # computes the reply text; play_reply() (called later by the
                # state machine) is the only place audio is produced.
                orch_result = harness.pipeline.process(result, source="microphone")
                current_event["facade"] = orch_result.event
                current_event["reply"] = {"text": orch_result.reply_text,
                                          "speak": True, "source": "facade"}
                return orch_result

            def _ducked_play(play_fn, *, kind: str):
                """Play ``play_fn()`` with the output ducked underneath.

                Mirrors real Echo/Alexa behaviour: while the assistant is
                producing audio (a spoken reply, an alarm, a timer ding) the
                master volume dips to :data:`DUCK_LEVEL` so the microphone can
                still catch the wake word, then returns to the pre-play level.
                Fail-soft -- if ducking is unavailable the audio still plays at
                full volume rather than failing.

                Also raises the shared ``tts_active`` flag for the whole
                playback window (including the blocking ``play()`` call) so the
                microphone loop closes the mic and drops our own voice. Without
                this the reply is heard by the mic, re-recognised as a second
                command, and surfaces as a spurious "could you repeat it?" plus
                a double reply.
                """
                nonlocal post_reply_until
                ctrl = volume_controller
                tts_active.set()
                if False:  # ducking disabled
                    try:
                        ctrl.duck()
                    except Exception:  # noqa: BLE001
                        log(f"[{kind}] duck failed; playing at full volume",
                            error=True)
                try:
                    return play_fn()
                finally:
                    if False:  # ducking disabled
                        try:
                            ctrl.unduck()
                        except Exception:  # noqa: BLE001
                            log(f"[{kind}] unduck failed", error=True)
                    # Post-reply tail: keep the mic closed for a brief
                    # additional window so speaker ring-down and early room
                    # reflections are absorbed before the wake detector
                    # re-arms. 300 ms is enough for a small-room speaker to
                    # settle without noticeably delaying the next interaction.
                    time.sleep(0.3)
                    tts_active.clear()
                    # Post-reply guard: the room keeps echoing the reply for a
                    # couple of seconds after the speaker settles. Hold the mic
                    # closed for post_reply_guard_s more so that lingering echo
                    # cannot re-latch the wake detector (the phantom "Alexa
                    # detected" + OOV reply). Applies to every reply, including
                    # the OOV reply itself, which breaks the echo loop.
                    if post_reply_guard_s > 0:
                        post_reply_until = time.monotonic() + post_reply_guard_s

            def play_reply(result, action):
                execution = getattr(action, "execution", None)
                payload = getattr(execution, "payload", {}) or {}
                if (result.get("intent") == "play_music"
                        and payload.get("announced_before_play")):
                    current_event["tts"] = {
                        "played": True,
                        "source": "piper",
                        "text": getattr(action, "reply_text", ""),
                    }
                    return

                if (result.get("intent") == "what_reminders" and piper_tts
                        and wav_player and payload.get("answer_parts")):
                    segments = payload["answer_parts"]
                    temp_paths = []
                    try:
                        for segment in segments:
                            with tempfile.NamedTemporaryFile(
                                    prefix="pi5-vcm-reminder-", suffix=".wav",
                                    delete=False) as temp:
                                temp_paths.append(temp.name)
                            piper_tts.synthesize(segment, temp_paths[-1])
                        playback = []
                        for index, path in enumerate(temp_paths):
                            preroll_ms = 750 if index == 0 else 0
                            playback.append(_ducked_play(
                                lambda path=path, preroll_ms=preroll_ms:
                                    wav_player.play(path, preroll_ms=preroll_ms),
                                kind="tts"))
                            if index + 1 < len(temp_paths):
                                time.sleep(0.5)
                        current_event["tts"] = {
                            "played": True,
                            "source": "piper",
                            "text": getattr(action, "reply_text", ""),
                            "segments": len(temp_paths),
                            "inter_item_pause_ms": 800,
                            "playback": playback,
                        }
                    finally:
                        for path in temp_paths:
                            os.unlink(path)
                    return

                # Dynamic (Piper TTS) intents: the spoken value is computed at
                # runtime (clock, live weather, reminder list), so it must be
                # synthesized from text rather than played from a static WAV.
                # Everything else keeps the pre-baked reply WAV.
                # pause_music / stop_music are dynamic too: their responses
                # depend on playback state ("No music is playing.",
                # "Music paused", "Music stopped"), so Piper uses the live
                # reply rather than a static WAV.
                _dynamic_intents = ("what_time", "what_weather", "what_reminders",
                                    "pause_music", "stop_music", "set_timer")
                if result.get("intent") in _dynamic_intents and piper_tts:
                    # Prefer the facade/live reply already computed during
                    # infer_command (it carries the real weather line / clock /
                    # reminder list). Fall back to the static template only if
                    # no live reply text was produced.
                    reply_text = ""
                    if isinstance(current_event.get("reply"), dict):
                        reply_text = str(current_event["reply"].get("text") or "")
                    if not reply_text:
                        reply_text = build_reply(result, action)["text"]
                    temp = tempfile.NamedTemporaryFile(
                        prefix="pi5-vcm-dyn-", suffix=".wav", delete=False
                    )
                    temp.close()
                    try:
                        piper_tts.synthesize(reply_text, temp.name)
                        current_event["tts"] = _ducked_play(
                            lambda: wav_player.play(temp.name), kind="tts")
                        current_event["tts"]["source"] = "piper"
                        current_event["tts"]["text"] = reply_text
                    finally:
                        os.unlink(temp.name)
                    return

                wav_name = reply_wav_name(result, action)
                wav_path = Path(args.reply_dir) / wav_name
                if not wav_path.exists():
                    # Fail soft: a missing/corrupt reply asset must never crash
                    # the interaction loop. Log it and stay silent for this
                    # turn rather than raising out of the state machine.
                    log(f"[reply] reply WAV not found, staying silent: {wav_path}",
                        error=True)
                    current_event["tts"] = {"played": False, "error": "missing_wav"}
                    return
                try:
                    current_event["tts"] = _ducked_play(
                        lambda: wav_player.play(str(wav_path)), kind="tts")
                except Exception as exc:  # noqa: BLE001 - never kill the loop
                    log(f"[reply] reply playback failed ({wav_name}): {exc}",
                        error=True)
                    current_event["tts"] = {"played": False, "error": str(exc)}

            def play_ack():
                if args.no_ack:
                    return None
                ack_path = str(Path(args.ack_wav))
                # Play the acknowledgement *blocking* so it finishes before
                # command capture begins. The previous play_async() let the beep
                # run over the user's command, contaminating the VAD capture and
                # pushing the recognition below the confidence floor ("please
                # repeat"). The capture loop's post-ack cooldown then discards
                # the beep's tail. Fail-soft: a missing/failed ack must not kill
                # the interaction.
                #
                # AEC: raise tts_active for the duration of the ack beep so the
                # microphone loop drops every frame while the speaker is
                # producing the two-tone chirp. Without this the beep's own
                # energy (and its room reflection) reaches the mic and can
                # either trip the VAD or, worse, be mis-scored by the wake-word
                # detector as a second "Alexa" (the classic double-trigger).
                tts_active.set()
                try:
                    result = wav_player.play(ack_path)
                    log_debug(f"ack_finished wav={ack_path}")
                    return result
                except Exception as exc:  # noqa: BLE001
                    log(f"[ack] acknowledgement playback failed: {exc}",
                        error=True)
                    return None
                finally:
                    tts_active.clear()

            def on_command_speech() -> None:
                rgb.wake()
                if timer_alarm is not None:
                    timer_alarm.pause()

            def on_command_timeout() -> None:
                machine.cancel_listening()
                if timer_alarm is not None:
                    timer_alarm.resume()

            machine = HarnessStateMachine(
                rgb=rgb,
                play_ack=play_ack,
                infer=infer_command,
                act=execute_action,
                play_reply=play_reply,
            )

            def acknowledge_wake() -> None:
                log("[wake] Alexa detected")
                log_debug("wake_detected")
                if detector is None:
                    return
                # Clear the wake detector BEFORE acknowledging. The detector
                # keeps a rolling buffer of the audio that just triggered it
                # (the user's own "Alexa"); if we ack first and the state
                # machine early-returns (already LISTENING), that buffer would
                # linger and re-latch on the next window, firing a spurious
                # second wake ("two Alexas"). Resetting first guarantees every
                # wake -- including a re-wake mid-listen -- starts clean.
                detector.reset()
                log("[acknowledging] acknowledgement playback skipped"
                    if args.no_ack else "[acknowledging] playing acknowledgement")
                machine.acknowledge_wake()
                log("[listening] acknowledgement complete")
                log_debug("listening_entered")

            try:
                for wav in microphone_utterances(
                    config=vad_config,
                        wakeword=detector,
                        on_wake=acknowledge_wake,
                        on_speech=on_command_speech,
                        on_timeout=on_command_timeout,
                        tts_active=tts_active.is_set,
                        post_reply_gate=(lambda: time.monotonic() < post_reply_until),
                        command_timeout_s=machine.command_timeout_s,
                        wake_only=args.wake_only):
                    if args.wake_only:
                        continue
                    # Stamp the command with the interaction's generation so a
                    # command captured by an overlapping wake is dropped instead
                    # of firing a second, simultaneous reply.
                    outcome = machine.handle_command(
                        wav, generation=machine.current_generation())
                    if timer_alarm is not None:
                        timer_alarm.resume()
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
        if media_player is not None:
            try:
                media_player.stop()
                log("[shutdown] music stopped")
            except Exception as exc:  # noqa: BLE001
                log(f"[shutdown] music stop failed: {exc}", error=True)
        if volume_controller is not None:
            try:
                volume_controller.end()
                log("[shutdown] output volume restored to pre-demo level")
            except Exception as exc:  # noqa: BLE001
                log(f"[shutdown] volume restore failed: {exc}", error=True)
        timer_manager.close()
        if timer_alarm is not None:
            timer_alarm.stop()
        rgb.close()

    for event in events:
        print(event_json(event))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
