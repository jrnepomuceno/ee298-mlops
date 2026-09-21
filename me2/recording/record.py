#!/usr/bin/env python3
"""Recording-session tool for the ME2 Voice Command Model (VCM).

Captures real 16 kHz mono speech from a microphone (e.g. a USB HyperX mic)
and writes WAV files plus a training manifest that plugs straight into::

    python main.py train --manifest recording/data/manifest.jsonl

The phrases you are prompted to say come from the SAME templates the
synthetic generator uses (``dataset.py``), so the recordings match the
command set and slot vocabulary the model is built for. No torch needed for
the audio path -- only ``sounddevice`` + ``soundfile`` (+ numpy).

Quick start
-----------
    pip install -r requirements.txt
    python record.py --list-devices          # find your mic
    python record.py --speaker joven         # start a recording session
    python record.py --status                # show coverage so far

Output (under ``recording/data/``)::

    data/<speaker>/<intent>_<n>.wav          # 16 kHz mono PCM-16
    data/manifest.jsonl                      # path, intent, transcript, speaker
    data/manifest.csv                        # same, as CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np

# --- project layout (paths are relative to the me2/ project root) ---------
REC_DIR = Path(__file__).resolve().parent          # .../me2/recording
PROJECT_ROOT = REC_DIR.parent                      # .../me2
DATA_DIR = REC_DIR / "data"

# --- audio settings (mirror config.py / the plan's recording spec) --------
SAMPLE_RATE = 16_000          # Hz, mono (config.SAMPLE_RATE)
PRE_SILENCE_S = 0.3           # leading silence to prepend  (plan: 0.3-0.5 s)
POST_SILENCE_S = 0.4          # trailing silence to append  (plan: 0.3-0.5 s)
MAX_UTTERANCE_S = 5.0         # hard cap per clip
SILENCE_STOP_S = 0.5          # stop after this much trailing quiet
ONSET_FACTOR = 3.0            # speech = RMS > noise_floor * this
ABS_FLOOR = 0.004             # absolute RMS floor (quiet-room safe)

MANIFEST_FIELDS = ["path", "intent", "transcript", "speaker"]

# --------------------------------------------------------------------------
# Templates. Primary source of truth is dataset.py (single source of truth,
# so recordings always match the synthetic generator). The bundled fallback
# below is a verbatim copy used only if dataset.py (and its torch import)
# is unavailable, so the recorder still runs on a bare machine.
# --------------------------------------------------------------------------
_FALLBACK_TEMPLATES: list[tuple[str, list[str]]] = [
    ("turn_on_lights", ["turn on the lights", "turn on the light",
                        "please turn on the lights"]),
    ("turn_off_lights", ["turn off the lights", "turn off the light",
                         "please turn off the lights"]),
    ("dim_lights", ["dim the lights to {pct} percent",
                    "set the lights to {pct} percent",
                    "dim the lights to {pct}"]),
    ("set_temperature", ["set the temperature to {num} degrees",
                         "set temperature to {num} degrees"]),
    ("play_music", ["play music", "play some music", "play music please"]),
    ("pause_music", ["pause the music", "pause music"]),
    ("stop_music", ["stop the music", "stop music"]),
    ("set_timer", ["set a timer for {num} minutes",
                   "set timer for {num} minutes",
                   "set a timer for {num} seconds"]),
    ("set_alarm", ["set an alarm for {num} {ampm}",
                   "set alarm for {num} {ampm}"]),
    ("cancel_timer", ["cancel the timer", "cancel timer", "stop the timer"]),
    ("remind", ["remind me to water the plants", "remind me to buy milk",
                "remind me to buy bread", "remind me to buy eggs",
                "remind me about the meeting", "remind me about the doctor",
                "remind me about the gym", "remind me about homework"]),
    ("call", ["call {contact}", "call {contact} please"]),
    ("what_time", ["what time is it", "what is the time"]),
    ("what_weather", ["what is the weather", "what is the weather like",
                      "how is the weather"]),
    ("what_reminders", ["what are my reminders", "show my reminders"]),
]
_FALLBACK_OOV: list[str] = [
    "tell me a joke", "who are you", "sing a song",
    "what is the meaning of life", "read the news",
    "how are you doing today", "tell me a story",
    "what is two plus two", "open the window",
    "close the door", "i am hungry", "good morning",
]
_FALLBACK_CONTACTS = ["mom", "dad", "john", "jane", "maria", "carlos"]


def _fallback_fill(template: str, rng: random.Random) -> str:
    import re
    def repl(m):
        key = m.group(1)
        if key == "pct":
            return str(rng.choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
        if key == "num":
            return str(rng.choice([1, 2, 3, 5, 10, 15, 20, 30, 45, 60]))
        if key == "ampm":
            return rng.choice(["am", "pm"])
        if key == "contact":
            return rng.choice(_FALLBACK_CONTACTS)
        return m.group(0)
    return re.sub(r"\{(\w+)\}", repl, template)


def _load_templates():
    """Return (TEMPLATES, OOV_TEMPLATES, CONTACTS, fill_fn) from dataset.py,
    falling back to the bundled copy if the import fails (no torch)."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from dataset import TEMPLATES, OOV_TEMPLATES, CONTACTS, _fill_template
        return TEMPLATES, OOV_TEMPLATES, CONTACTS, _fill_template
    except Exception as e:  # noqa: BLE001 - any import failure -> fallback
        print(f"[warn] could not import dataset.py ({e}); "
              f"using bundled template copy", file=sys.stderr)
        return _FALLBACK_TEMPLATES, _FALLBACK_OOV, _FALLBACK_CONTACTS, \
            _fallback_fill


# --------------------------------------------------------------------------
# Phrase generation
# --------------------------------------------------------------------------
_SINGULAR = {
    "1 seconds": "1 second", "1 minutes": "1 minute",
    "1 hours": "1 hour", "1 degrees": "1 degree",
}


def _normalize(phrase: str) -> str:
    """Fix ungrammatical '1 <plural>' so the prompt matches natural speech
    (and thus the transcript the speaker actually says)."""
    for bad, good in _SINGULAR.items():
        phrase = phrase.replace(bad, good)
    return phrase


def build_phrase_pool(intent: str, templates, oov, contacts, fill, n: int,
                      rng: random.Random) -> list[str]:
    """Return ~n phrases for an intent, filling slot templates with varied
    values (different numbers / am-pm / contacts) for natural variation."""
    if intent == "oov":
        base = list(oov)
    else:
        base = [t for (i, ts) in templates if i == intent for t in ts]
    if not base:
        return []
    per = max(1, n // len(base))
    pool = [_normalize(fill(t, rng)) for t in base for _ in range(per)]
    rng.shuffle(pool)
    return pool


def speaker_rng(intent: str, speaker: str) -> random.Random:
    """Stable, per-(intent, speaker) RNG so phrases are reproducible and
    differ across speakers (crc32 is not salted like hash())."""
    seed = zlib.crc32(f"{intent}|{speaker}".encode())
    return random.Random(seed)


# --------------------------------------------------------------------------
# Audio I/O (sounddevice / soundfile imported lazily)
# --------------------------------------------------------------------------
def _sd():
    try:
        import sounddevice as sd
        return sd
    except Exception as e:  # noqa: BLE001
        sys.exit("sounddevice is not installed. Run:\n"
                 "  pip install -r recording/requirements.txt\n"
                 f"(import error: {e})")


def _sf():
    try:
        import soundfile as sf
        return sf
    except Exception as e:  # noqa: BLE001
        sys.exit("soundfile is not installed. Run:\n"
                 "  pip install -r recording/requirements.txt\n"
                 f"(import error: {e})")


def list_devices() -> None:
    sd = _sd()
    print(sd.query_devices())
    default_in = sd.default.device[0]
    print(f"\nDefault input device index: {default_in}")
    print("Pass it with --device <index>.")


def beep(sd, freq: float = 880.0, dur: float = 0.2, vol: float = 0.3) -> None:
    t = np.arange(int(SAMPLE_RATE * dur)) / SAMPLE_RATE
    tone = (vol * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sd.play(tone, SAMPLE_RATE)
    sd.wait()


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def record_utterance(sd, device=None, fixed: float | None = None,
                     max_s: float = MAX_UTTERANCE_S,
                     silence_s: float = SILENCE_STOP_S,
                     onset_factor: float = ONSET_FACTOR,
                     abs_floor: float = ABS_FLOOR,
                     block: int = 512):
    """Record one utterance with energy-based VAD.

    Returns (wav: np.ndarray float32 | None, speech_duration_s: float).
    ``wav`` is None when no speech is detected. The returned clip already
    has PRE_SILENCE_S prepended and POST_SILENCE_S appended.
    """
    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].astype(np.float32).copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        device=device, blocksize=block, callback=cb):
        # --- noise floor from the first ~0.3 s ----------------------------
        nf_blocks = max(1, int(0.3 * SAMPLE_RATE / block))
        nf = [_rms(q.get()) for _ in range(nf_blocks)]
        noise = max(float(np.median(nf)), abs_floor)
        thr = max(noise * onset_factor, abs_floor * 2.0)

        if fixed is not None:
            # Fixed-duration capture (foolproof alternative to VAD).
            n = int(fixed * SAMPLE_RATE)
            chunks = []
            got = 0
            while got < n:
                c = q.get()
                chunks.append(c)
                got += len(c)
            speech = np.concatenate(chunks)[:n]
            pre = np.zeros(int(PRE_SILENCE_S * SAMPLE_RATE), np.float32)
            post = np.zeros(int(POST_SILENCE_S * SAMPLE_RATE), np.float32)
            return np.concatenate([pre, speech, post]), float(fixed)

        # --- VAD state machine -------------------------------------------
        buf: list[np.ndarray] = []
        speech_start: int | None = None
        low_frames = 0
        max_frames = int(max_s * SAMPLE_RATE)
        silence_frames = int(silence_s * SAMPLE_RATE)

        while True:
            c = q.get()
            buf.append(c)
            total = sum(len(x) for x in buf)
            r = _rms(c)
            if speech_start is None:
                if r > thr:
                    speech_start = len(buf) - 1
                    low_frames = 0
            else:
                low_frames = low_frames + len(c) if r < thr * 0.6 else 0
                if low_frames >= silence_frames or total >= max_frames:
                    break
            if total >= max_frames:
                break

    if speech_start is None:
        return None, 0.0
    speech = np.concatenate(buf[speech_start:])
    pre = np.zeros(int(PRE_SILENCE_S * SAMPLE_RATE), np.float32)
    post = np.zeros(int(POST_SILENCE_S * SAMPLE_RATE), np.float32)
    return np.concatenate([pre, speech, post]), float(len(speech) / SAMPLE_RATE)


def save_wav(path: Path, wav: np.ndarray) -> None:
    sf = _sf()
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, SAMPLE_RATE, subtype="PCM_16")


# --------------------------------------------------------------------------
# Manifest + progress
# --------------------------------------------------------------------------
def load_manifest_state(manifest_path: Path):
    """Return (entries: list[dict], seen: set[(speaker,intent,transcript)])."""
    entries: list[dict] = []
    if manifest_path.suffix == ".jsonl" and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
    elif manifest_path.suffix == ".csv" and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            entries.extend(list(csv.DictReader(f)))
    seen = {(e["speaker"], e["intent"], e["transcript"]) for e in entries}
    return entries, seen


def write_manifest(manifest_path: Path, entries: list[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    csv_path = manifest_path.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for e in entries:
            w.writerow({k: e.get(k, "") for k in MANIFEST_FIELDS})


def count_intent(entries: list[dict], intent: str) -> int:
    return sum(1 for e in entries if e["intent"] == intent)


def count_intent_speaker(entries: list[dict], intent: str, speaker: str) -> int:
    return sum(1 for e in entries
               if e["intent"] == intent and e["speaker"] == speaker)


def print_status(entries: list[dict], templates) -> None:
    intents = [i for (i, _) in templates] + ["oov"]
    speakers = sorted({e["speaker"] for e in entries})
    print(f"\nManifest coverage  ({len(entries)} clips, "
          f"{len(speakers)} speaker(s))\n")
    print(f"{'intent':<18}{'total':>7}   per-speaker")
    print("-" * 54)
    for it in intents:
        tot = count_intent(entries, it)
        per = ", ".join(f"{s}={count_intent_speaker(entries, it, s)}"
                        for s in speakers if count_intent_speaker(entries, it, s))
        print(f"{it:<18}{tot:>7}   {per or '-'}")
    print("-" * 54)


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------
def run_session(args, templates, oov, contacts, fill) -> None:
    sd = _sd()
    manifest_path = Path(args.manifest)
    entries, seen = load_manifest_state(manifest_path)

    work = (args.intent.split(",") if args.intent
            else [i for (i, _) in templates] + ["oov"])
    work = [w.strip() for w in work if w.strip()]

    print(f"\n=== Recording session: speaker='{args.speaker}' ===")
    print(f"Target: {args.target} clips/intent (total), "
          f"max {args.per_speaker}/intent for this speaker.")
    print(f"Intents this session: {len(work)}")
    print("Keys: [Enter]=record  r=retry  s=skip intent  q=quit\n")

    for intent in work:
        total = count_intent(entries, intent)
        mine = count_intent_speaker(entries, intent, args.speaker)
        if total >= args.target and not args.force:
            print(f"[{intent}] {total}/{args.target} total -- done, "
                  f"skipping (use --force to add more)")
            continue
        pool = build_phrase_pool(intent, templates, oov, contacts, fill,
                                 max(args.per_speaker, 4),
                                 speaker_rng(intent, args.speaker))
        if not pool:
            print(f"[{intent}] no phrases -- skipping")
            continue
        pi = 0
        while mine < args.per_speaker and total < args.target:
            phrase = pool[pi % len(pool)]
            pi += 1
            print(f"[{intent}]  total {total}/{args.target}   "
                  f"you {mine}/{args.per_speaker}")
            print(f"  Say:  {phrase}")
            try:
                choice = input("  [Enter=record, r=retry, s=skip, q=quit] "
                               ).strip().lower()
            except EOFError:
                break
            if choice == "q":
                return
            if choice == "s":
                break
            if choice == "r":
                continue  # re-show the same phrase

            try:
                beep(sd)
                wav, dur = record_utterance(
                    sd, device=args.device, fixed=args.fixed,
                    max_s=args.max_s, silence_s=args.silence,
                    onset_factor=args.onset, abs_floor=args.floor)
            except KeyboardInterrupt:
                print("\n  (interrupted)")
                continue
            if wav is None:
                print("  (no speech detected -- try again)")
                continue

            n = mine + 1
            rel = f"recording/data/{args.speaker}/{intent}_{n:03d}.wav"
            save_wav(PROJECT_ROOT / rel, wav)
            key = (args.speaker, intent, phrase)
            dup = key in seen
            entries.append({"path": rel, "intent": intent,
                            "transcript": phrase, "speaker": args.speaker})
            seen.add(key)
            total += 1
            mine += 1
            peak = float(np.abs(wav).max())
            flag = "  (repeat phrase)" if dup else ""
            print(f"  saved {rel}  ({dur:.2f}s speech, peak {peak:.2f}){flag}")

    write_manifest(manifest_path, entries)
    print(f"\nWrote {len(entries)} entries -> {manifest_path}")
    print_status(entries, templates)
    print("\nTip: have the next speaker run with --speaker <name>, then "
          "train with:\n  python main.py train --manifest "
          "recording/data/manifest.jsonl")


def dry_run(args, templates, oov, contacts, fill) -> None:
    work = (args.intent.split(",") if args.intent
            else [i for (i, _) in templates] + ["oov"])
    print(f"\nDry run -- phrases that would be recorded "
          f"(speaker='{args.speaker}', {args.per_speaker}/intent):\n")
    for intent in [w.strip() for w in work if w.strip()]:
        pool = build_phrase_pool(intent, templates, oov, contacts, fill,
                                 max(args.per_speaker, 4),
                                 speaker_rng(intent, args.speaker))
        print(f"{intent} ({len(pool)} phrases):")
        for p in pool:
            print(f"   - {p}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Record real speech for the VCM training set.")
    ap.add_argument("--speaker", type=str, default=None,
                    help="speaker id, e.g. 'joven' (required to record)")
    ap.add_argument("--device", type=int, default=None,
                    help="input device index (see --list-devices)")
    ap.add_argument("--intent", type=str, default=None,
                    help="comma-separated intents to record (default: all)")
    ap.add_argument("--target", type=int, default=30,
                    help="target clips per intent, total across speakers "
                         "(default 30)")
    ap.add_argument("--per-speaker", type=int, default=6,
                    help="max clips this speaker records per intent "
                         "(default 6; 5 speakers x 6 = 30)")
    ap.add_argument("--fixed", type=float, default=None,
                    help="record a fixed number of seconds instead of VAD")
    ap.add_argument("--max-s", type=float, default=MAX_UTTERANCE_S,
                    help=f"VAD hard cap in seconds (default {MAX_UTTERANCE_S})")
    ap.add_argument("--silence", type=float, default=SILENCE_STOP_S,
                    help=f"trailing quiet (s) that ends a clip "
                         f"(default {SILENCE_STOP_S})")
    ap.add_argument("--onset", type=float, default=ONSET_FACTOR,
                    help=f"speech onset = noise floor x this "
                         f"(default {ONSET_FACTOR})")
    ap.add_argument("--floor", type=float, default=ABS_FLOOR,
                    help=f"absolute RMS floor (default {ABS_FLOOR})")
    ap.add_argument("--manifest", type=str,
                    default=str(DATA_DIR / "manifest.jsonl"),
                    help="manifest path (JSONL or CSV)")
    ap.add_argument("--force", action="store_true",
                    help="record even if the intent already meets --target")
    ap.add_argument("--list-devices", action="store_true",
                    help="list audio input devices and exit")
    ap.add_argument("--status", action="store_true",
                    help="print manifest coverage and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the phrases that would be recorded, no mic")
    args = ap.parse_args()

    templates, oov, contacts, fill = _load_templates()

    if args.list_devices:
        list_devices()
        return
    if args.status:
        entries, _ = load_manifest_state(Path(args.manifest))
        print_status(entries, templates)
        return
    if args.dry_run:
        dry_run(args, templates, oov, contacts, fill)
        return

    if not args.speaker:
        ap.error("--speaker is required to record (or use --list-devices / "
                 "--status / --dry-run)")
    run_session(args, templates, oov, contacts, fill)


if __name__ == "__main__":
    main()
