#!/usr/bin/env python3
"""Regenerate every canned reply WAV for the Pi5 voice assistant.

Two families of assets live in ``assets/replies/``:

1. **Spoken replies** -- one per intent, produced by ``espeak-ng`` so the whole
   set shares a single, consistent voice. These are the deterministic, offline
   answers the assistant plays when a command is recognised (no LLM, no cloud).
   The spoken text mirrors the live templates in ``rpi5/facade.py`` / the
   executor answers, so the static WAV and the dynamic (Piper) reply agree.

2. **Beeps / tones** -- the WILLEN priming beep, the wake acknowledgement, and
   the timer/alarm expiry sounds, synthesised with pure stdlib (math/wave) so
   they are byte-reproducible with no external dependency.

Why regenerate
--------------
``assets/replies/willen_prime.wav`` had been committed as a *completely silent*
file (every frame zero), so the WILLEN priming beep was inaudible. And the
reply set had drifted out of sync with the intent taxonomy -- several intents
had no dedicated reply WAV and fell back to ``oov.wav``. This script rebuilds
the full, consistent set in one shot.

Format: 22050 Hz, mono, 16-bit PCM -- matching the original assets so the
Pi's ``pw-play`` / the Mac's ``afplay`` handle them identically.

Usage
-----
    python3 scripts/make_replies.py            # regenerate everything
    python3 scripts/make_replies.py --check    # validate only, write nothing
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import struct
import subprocess
import sys
import wave

SR = 22050          # sample rate (matches the original assets)
SAMPLE_WIDTH = 2    # 16-bit
CHANNELS = 1

HERE = os.path.dirname(os.path.abspath(__file__))
# run.py defaults are relative to the me2/ working dir (see run_pi5_harness.sh),
# so the live assets dir is me2/assets/replies/, not rpi5/assets/replies/.
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "assets", "replies"))

ESPEAK_VOICE = "en-us"          # single voice for the whole set -> consistency
ESPEAK_SPEED = 172              # words/minute (default 175); slightly slower = clearer


# --------------------------------------------------------------------------- #
# Spoken replies.
#
# intent -> spoken text. Static, slot-free phrasing that matches the live
# facade templates (rpi5/facade.py INTENT_SPECS) and the executor answers, so
# the canned WAV is what the assistant actually says. Dynamic intents (timer
# length, alarm time, temperature, contact, reminder note) keep their generic
# confirmation here; the live Piper path still speaks the slot-filled version.
# --------------------------------------------------------------------------- #
REPLY_TEXT = {
    "turn_on_lights":   "Turning the lights on.",
    "turn_off_lights":  "Turning the lights off.",
    "dim_lights":       "Dimming the lights.",
    "set_temperature":  "Setting the temperature.",
    "play_music":       "Playing music.",
    "pause_music":      "Pausing the music.",
    "stop_music":       "Stopping the music.",
    # (pause_music / stop_music are now spoken dynamically from the live reply
    #  -- "No music is playing." when idle -- so these canned WAVs are only a
    #  fallback if Piper TTS is unavailable.)
    "set_timer":        "Setting the timer.",
    "set_alarm":        "Alarm set.",
    "stop_timer":       "Timer stopped.",
    "remind":           "Reminder set.",
    "call":             "Calling.",
    "what_time":        "Checking the time.",
    "what_weather":     "Checking the weather.",
    "what_reminders":   "Reading your reminders.",
    "volume_up":        "Turning the volume up.",
    "volume_down":      "Turning the volume down.",
    "mute":             "Muting the output.",
    "oov":              "I did not recognise that command.",
}

# Every intent in config.INTENTS must have a spoken reply WAV. This is the
# completeness contract that was previously violated (volume_up/down, mute and
# stop_timer had no dedicated WAV and fell back to oov.wav).
REQUIRED_INTENTS = set(REPLY_TEXT)


# --------------------------------------------------------------------------- #
# Pure-stdlib tone synthesis (same style as scripts/make_expiry_sounds.py).
# --------------------------------------------------------------------------- #
def _env(t: float, dur: float, attack: float = 0.005, release: float = 0.04) -> float:
    """Linear attack/release envelope in [0, 1]."""
    if t < 0:
        return 0.0
    if t < attack:
        return t / attack
    rel_start = dur - release
    if t > rel_start:
        return max(0.0, (dur - t) / release)
    return 1.0


def _tone(freq: float, dur: float, sr: int = SR) -> list[float]:
    n = int(sr * dur)
    out = []
    for i in range(n):
        t = i / sr
        out.append(math.sin(2 * math.pi * freq * t) * _env(t, dur))
    return out


def _concat(parts: list[list[float]], gap: float = 0.0) -> list[float]:
    out = list(parts[0])
    gap_n = int(SR * gap)
    for p in parts[1:]:
        out.extend([0.0] * gap_n)
        out.extend(p)
    return out


def _normalize(samples: list[float], peak: float = 0.9) -> list[int]:
    m = max((abs(s) for s in samples), default=0.0)
    scale = peak / m if m > 0 else 0.0
    return [int(max(-1.0, min(1.0, s * scale)) * 32767) for s in samples]


def _write_wav(path: str, samples: list[int]) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SR)
        w.writeframes(struct.pack("<%dh" % len(samples), *samples))


# --------------------------------------------------------------------------- #
# Beep generators.
# --------------------------------------------------------------------------- #
def make_willen_prime() -> list[int]:
    """WILLEN priming beep: a single short, clear mid-high tone.

    Historically this file was committed all-zero (silent) -- the priming beep
    never sounded. A single 880 Hz (A5) tone with a soft envelope is unmissable
    and clearly distinct from the two-tone acknowledgement below.
    """
    samples = _tone(880.0, 0.14)
    return _normalize(samples, peak=0.7)


def make_ack_beep() -> list[int]:
    """Wake acknowledgement: a friendly two-note rise (E5 -> A5)."""
    samples = _concat([_tone(659.25, 0.09), _tone(880.0, 0.12)], gap=0.02)
    return _normalize(samples, peak=0.7)


def make_timer_ding() -> list[int]:
    """Countdown timer expiry: gentle rising triple-ding (C6 E6 G6)."""
    notes = [1046.50, 1318.51, 1568.00]
    samples = _concat([_tone(f, 0.16) for f in notes], gap=0.06)
    return _normalize(samples, peak=0.55)


def make_alarm_ring() -> list[int]:
    """Wall-clock alarm: four urgent low buzzes."""
    def buzz(freq: float, dur: float) -> list[float]:
        n = int(SR * dur)
        out = []
        for i in range(n):
            t = i / SR
            s = math.sin(2 * math.pi * freq * t) + 0.5 * math.sin(4 * math.pi * freq * t)
            out.append((s / 1.5) * _env(t, dur, attack=0.003, release=0.02))
        return out
    samples = _concat([buzz(660.0, 0.28) for _ in range(4)], gap=0.12)
    return _normalize(samples, peak=0.9)


# --------------------------------------------------------------------------- #
# espeak-ng synthesis.
# --------------------------------------------------------------------------- #
def _find_espeak() -> str | None:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        for cand in ("/opt/homebrew/bin/espeak-ng", "/usr/bin/espeak-ng"):
            if os.path.exists(cand):
                exe = cand
                break
    return exe


def synth_reply(espeak: str, text: str, dest: str) -> bool:
    """Synthesise one reply WAV with espeak-ng at the target sample rate."""
    tmp = dest + ".tmp.wav"
    cmd = [espeak, "-v", ESPEAK_VOICE, "-s", str(ESPEAK_SPEED),
           "-w", tmp, text]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        print(f"  ! espeak failed for {text!r}: {exc}")
        return False
    # espeak-ng emits 22050 Hz mono 16-bit by default; verify and copy through.
    try:
        with wave.open(tmp, "rb") as w:
            if w.getframerate() != SR:
                # Resample to SR with a simple linear interpolation so the
                # asset matches the rest of the set exactly.
                raw = w.readframes(w.getnframes())
                samples = list(struct.unpack("<%dh" % (len(raw) // 2), raw))
                ratio = SR / w.getframerate()
                new_len = int(len(samples) * ratio)
                resampled = [int(samples[min(len(samples) - 1,
                                             int(i / ratio))]) for i in range(new_len)]
                _write_wav(dest, resampled)
            else:
                shutil.move(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return True



# --------------------------------------------------------------------------- #
# Piper TTS synthesis (preferred when espeak-ng is unavailable).
# --------------------------------------------------------------------------- #
PIPER_BIN_DEFAULT = os.path.expanduser("~/piper-venv/bin/piper")
PIPER_MODEL_DEFAULT = os.path.expanduser("~/piper-voices/en_US-lessac-medium.onnx")


def _find_piper():
    """Locate the Piper binary and voice model. Returns (bin, model) or None."""
    bin_path = shutil.which("piper") or (
        PIPER_BIN_DEFAULT if os.path.isfile(PIPER_BIN_DEFAULT) else None
    )
    model_path = PIPER_MODEL_DEFAULT if os.path.isfile(PIPER_MODEL_DEFAULT) else None
    if bin_path and model_path:
        return bin_path, model_path
    return None


def synth_reply_piper(piper_bin, model, text, dest):
    """Synthesise one reply WAV with Piper. Resamples to SR if needed."""
    tmp = dest + ".tmp.wav"
    cmd = [piper_bin, "--model", model, "--output_file", tmp]
    try:
        subprocess.run(cmd, input=text + "\n", text=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       check=True, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        print(f"  ! piper failed for {text!r}: {exc}")
        return False
    try:
        with wave.open(tmp, "rb") as w:
            if w.getframerate() != SR:
                raw = w.readframes(w.getnframes())
                samples = list(struct.unpack("<%dh" % (len(raw) // 2), raw))
                ratio = SR / w.getframerate()
                new_len = int(len(samples) * ratio)
                resampled = [int(samples[min(len(samples) - 1, int(i / ratio))])
                             for i in range(new_len)]
                _write_wav(dest, resampled)
            else:
                shutil.move(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return True


# --------------------------------------------------------------------------- #
# Validation.
# --------------------------------------------------------------------------- #
def _peak_rms(path: str) -> tuple[int, float]:
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
    a = struct.unpack("<%dh" % (len(raw) // 2), raw)
    peak = max((abs(x) for x in a), default=0)
    rms = (sum(x * x for x in a) / len(a)) ** 0.5 if a else 0.0
    return peak, rms


def validate(out_dir: str) -> bool:
    """Confirm every required reply exists and is audibly non-silent."""
    ok = True
    print(f"\nValidating {out_dir}")
    for intent in sorted(REQUIRED_INTENTS):
        path = os.path.join(out_dir, f"{intent}.wav")
        if not os.path.exists(path):
            print(f"  MISSING  {intent}.wav")
            ok = False
            continue
        peak, rms = _peak_rms(path)
        verdict = "ok" if peak >= 500 else ("QUIET" if peak >= 50 else "SILENT!")
        if peak < 500:
            ok = False
        print(f"  {intent:18} peak={peak:6} rms={rms:7.1f}  {verdict}")
    # Beeps / tones (non-intent assets) must also be non-silent.
    for name in ("willen_prime", "ack_beep", "timer_ding", "alarm_ring"):
        path = os.path.join(out_dir, f"{name}.wav")
        if not os.path.exists(path):
            print(f"  MISSING  {name}.wav")
            ok = False
            continue
        peak, _ = _peak_rms(path)
        verdict = "ok" if peak >= 500 else "SILENT!"
        if peak < 500:
            ok = False
        print(f"  {name:18} peak={peak:6}  {verdict}")
    return ok


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="validate existing assets only; write nothing")
    ap.add_argument("--out", default=OUT_DIR, help="output directory")
    args = ap.parse_args()
    out_dir = args.out

    if args.check:
        return 0 if validate(out_dir) else 1

    os.makedirs(out_dir, exist_ok=True)

    # 1. Spoken replies (Piper preferred, espeak-ng fallback).
    piper = _find_piper()
    espeak = _find_espeak()
    if piper is not None:
        print(f"Synthesising {len(REQUIRED_INTENTS)} spoken replies with Piper")
        for intent in sorted(REQUIRED_INTENTS):
            dest = os.path.join(out_dir, f"{intent}.wav")
            if synth_reply_piper(piper[0], piper[1], REPLY_TEXT[intent], dest):
                print(f"  wrote {os.path.basename(dest)}")
    elif espeak is not None:
        print(f"Synthesising {len(REQUIRED_INTENTS)} spoken replies with espeak-ng")
        for intent in sorted(REQUIRED_INTENTS):
            dest = os.path.join(out_dir, f"{intent}.wav")
            if synth_reply(espeak, REPLY_TEXT[intent], dest):
                print(f"  wrote {os.path.basename(dest)}")
    else:
        print("WARNING: neither Piper nor espeak-ng found; skipping spoken replies.")
        print("         Install Piper (~/piper-venv) or espeak-ng and re-run.")

    # 2. Beeps / tones (pure stdlib).
    print("Synthesising beeps / tones (stdlib)")
    for name, fn in (("willen_prime", make_willen_prime),
                     ("ack_beep", make_ack_beep),
                     ("timer_ding", make_timer_ding),
                     ("alarm_ring", make_alarm_ring)):
        dest = os.path.join(out_dir, f"{name}.wav")
        _write_wav(dest, fn())
        print(f"  wrote {os.path.basename(dest)}")

    print()
    return 0 if validate(out_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
