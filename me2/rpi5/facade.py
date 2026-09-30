"""Model-facing facade: turn a raw recognition result into an action request.

The model (ONNX, torch-free) emits exactly three things: an ``intent`` string,
an ``intent_confidence``, and a small ``slots`` dict (see :mod:`model.slots`).
This module is the *only* place that decides what to do with them. It is pure
and synchronous -- no I/O, no hardware, no network -- so it is trivially unit
testable on the Pi or on a dev box.

Pipeline position::

    model  ->  facade.decode()  ->  ActionRequest | RejectResult  ->  orchestrator

Design rules
------------
* **The model stays dumb.** It never executes anything; it only names an
  intent and fills slots. Adding a new intent touches ``INTENT_SPECS`` here
  (and the model's label set) -- never the executors unless the category is
  new.
* **One validation table.** Every intent declares its required slots, their
  types, and sane ranges in :data:`INTENT_SPECS`. The gate and the validator
  both read from it, so a slot can't be "required" in one place and "optional"
  in another.
* **Deterministic replies.** Reply text is templated from slots (no LLM). The
  template lives next to the spec so intent, its slots, and its wording stay
  in one screen.
* **Reject, don't guess.** Anything below the confidence floor, any OOV, any
  missing/invalid slot is a :class:`RejectResult` with a reason -- never a
  half-applied action.

Deployment target is the Raspberry Pi 5: this file imports only the standard
library (no numpy, no torch, no onnxruntime).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ActionRequest:
    """A validated, executable command produced by :func:`decode`."""

    intent: str
    category: str                 # routing key for the orchestrator
    slots: Mapping[str, Any]      # normalized slot values
    action_code: str              # stable machine name, e.g. "lights.dim"
    reply_text: str               # what the assistant says back
    dry_run: bool = True          # True until hardware executors are wired
    confidence: float = 0.0


@dataclass(frozen=True)
class RejectResult:
    """Why a command was not executed, plus what to say instead."""

    reason: str                   # "oov"|"low_confidence"|"missing_slot"|"invalid_slot"
    detail: str
    reply_text: str
    intent: str = ""


# --------------------------------------------------------------------------- #
# Slot specs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SlotSpec:
    name: str
    kind: str                     # "int" | "str"
    required: bool = True
    lo: int | None = None         # inclusive lower bound (int slots)
    hi: int | None = None         # inclusive upper bound (int slots)


def _TIMER_REPLY(slots: Mapping[str, Any]) -> str:
    """Slot-aware timer reply (handles optional unit)."""
    dur = slots.get("duration")
    unit = str(slots.get("duration_unit", "minute")).rstrip("s")
    unit_word = "second" if unit.startswith("sec") else "minute"
    if dur is None:
        return "Setting the timer."
    return f"Timer set for {dur} {unit_word}s."


#: Per-intent contract: category, machine action code, required slots, reply
#: template (``{slot}`` placeholders). ``None`` action_code marks pure queries
#: that still produce a spoken answer.
INTENT_SPECS: dict[str, dict[str, Any]] = {
    # --- lights ----------------------------------------------------------- #
    "turn_on_lights":  dict(category="lights", code="lights.on",
                            slots=(), reply="Turning the lights on."),
    "turn_off_lights": dict(category="lights", code="lights.off",
                            slots=(), reply="Turning the lights off."),
    "dim_lights":      dict(category="lights", code="lights.dim",
                            slots=(SlotSpec("percent", "int", lo=1, hi=100),),
                            reply="Dimming the lights to {percent} percent."),
    # --- hvac ------------------------------------------------------------- #
    "set_temperature": dict(category="hvac", code="thermostat.set",
                            slots=(SlotSpec("temperature", "int", lo=10, hi=35),),
                            reply="Temperature set to {temperature} degrees."),
    # --- media ------------------------------------------------------------ #
    "play_music":  dict(category="media", code="media.play",
                        slots=(), reply="Playing music."),
    "pause_music": dict(category="media", code="media.pause",
                        slots=(), reply="Pausing the music."),
    "stop_music":  dict(category="media", code="media.stop",
                        slots=(), reply="Stopping the music."),
    # --- timers & alarms -------------------------------------------------- #
    "set_timer":   dict(category="timer", code="timer.set",
                        slots=(SlotSpec("duration", "int", lo=1, hi=1440),),
                        reply=_TIMER_REPLY),
    "set_alarm":   dict(category="timer", code="alarm.set",
                        slots=(SlotSpec("time", "str"),),
                        reply="Alarm set for {time}."),
    "stop_timer":  dict(category="timer", code="timer.cancel",
                        slots=(), reply="Stopping the timer."),
    # --- reminders -------------------------------------------------------- #
    "remind":      dict(category="remind", code="reminder.create",
                        slots=(SlotSpec("note", "str"),),
                        reply="I'll remind you to {note}."),
    # --- communications --------------------------------------------------- #
    "call":        dict(category="comms", code="call.request",
                        slots=(SlotSpec("contact", "str"),),
                        reply="Calling {contact}."),
    # --- output volume ---------------------------------------------------- #
    "volume_up":   dict(category="volume", code="volume.up",
                        slots=(), reply="Turning the volume up."),
    "volume_down": dict(category="volume", code="volume.down",
                        slots=(), reply="Turning the volume down."),
    # --- information (local / web) --------------------------------------- #
    "what_time":      dict(category="info", code="query.time",
                           slots=(), reply="The time is {now}."),
    "what_weather":   dict(category="info", code="query.weather",
                           slots=(), reply="Checking the weather."),
    "what_reminders": dict(category="info", code="query.reminders",
                           slots=(), reply="Reading your reminders."),
}

#: Intents the model is trained to emit (excluding oov). Kept in sync with
#: ``config.INTENTS``; used for the "known intent" check in the gate.
KNOWN_INTENTS: frozenset[str] = frozenset(INTENT_SPECS)

#: Confidence floor. Below this the model is not trusted to act.
DEFAULT_CONFIDENCE_THRESHOLD = 0.75

#: Human-readable slot labels for error messages ("the percent slot").
_SLOT_ARTICLE = {"percent": "the brightness", "temperature": "the temperature",
                 "duration": "the duration", "time": "the time",
                 "contact": "the contact", "note": "the reminder text"}


# --------------------------------------------------------------------------- #
# Normalization helpers (pure)
# --------------------------------------------------------------------------- #
_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
}


def _to_int(value: Any) -> int | None:
    """Best-effort int coercion for slot values.

    Handles ints, numeric strings ("40"), and single/common number words
    ("forty", "five hundred"). Returns ``None`` when it cannot coerce.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if float(value).is_integer() else None
    if not isinstance(value, str):
        return None
    s = value.strip().lower()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    words = s.replace("-", " ").split()
    total = 0
    seen = False
    for w in words:
        if w not in _WORD_NUMBERS:
            return None
        seen = True
        if w == "hundred":
            total = (total or 1) * 100
        else:
            total += _WORD_NUMBERS[w]
    return total if seen else None


def _clean_str(value: Any) -> str:
    return str(value).strip()


# --------------------------------------------------------------------------- #
# Reply rendering
# --------------------------------------------------------------------------- #
def _render(template: str | Any, slots: Mapping[str, Any], intent: str) -> str:
    """Fill a reply template.

    ``template`` is either a string with ``{slot}`` placeholders (``{now}`` is
    special-cased to the local time) or a zero-arg callable that receives the
    normalized slots (used by slot-shape-dependent replies like the timer).
    """
    if callable(template):
        return str(template(slots))
    if "{now}" in template:
        template = template.replace("{now}", _local_time_text())
    try:
        return template.format(**slots)
    except (KeyError, IndexError, ValueError):
        # A placeholder had no slot value, or a slot value contained a brace
        # ("The time is {slot}" where the slot itself is "{slot}"): fall back
        # to a generic line rather than crash the assistant mid-utterance.
        return f"I heard '{intent}'."


def _local_time_text() -> str:
    """Local wall-clock time as ``H:MM AM/PM`` (12-hour, no leading zero).

    Uses ``%I`` + manual strip instead of glibc-only ``%-I`` so the string is
    identical on the Pi (musl/glibc) and on any dev box.
    """
    now = datetime.now()
    hour12 = now.hour % 12 or 12
    return f"{hour12}:{now.strftime('%M')} {now.strftime('%p')}"


# --------------------------------------------------------------------------- #
# The gate + validator
# --------------------------------------------------------------------------- #
def _reject_oov(intent: str) -> RejectResult:
    return RejectResult(reason="oov", detail=f"intent '{intent}' is out of vocabulary",
                        reply_text="I don't understand.", intent=intent)


def _reject_low_conf(conf: float, threshold: float) -> RejectResult:
    return RejectResult(reason="low_confidence",
                        detail=f"confidence {conf:.3f} < {threshold:.3f}",
                        reply_text="I don't understand.",
                        intent="")


def _validate_slots(intent: str, slots: Mapping[str, Any]) -> tuple[dict[str, Any], str | None, str | None]:
    """Return (normalized_slots, missing_slot, invalid_detail).

    Exactly one of ``missing_slot`` / ``invalid_detail`` is set on failure.
    """
    spec = INTENT_SPECS[intent]
    out: dict[str, Any] = {}
    for ss in spec["slots"]:
        raw = slots.get(ss.name)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            label = _SLOT_ARTICLE.get(ss.name, f"the {ss.name}")
            return out, f"{label} is missing", None
        if ss.kind == "int":
            iv = _to_int(raw)
            if iv is None:
                return out, None, f"{ss.name}={raw!r} is not a number"
            if ss.lo is not None and iv < ss.lo:
                return out, None, f"{ss.name}={iv} is below {ss.lo}"
            if ss.hi is not None and iv > ss.hi:
                return out, None, f"{ss.name}={iv} is above {ss.hi}"
            out[ss.name] = iv
        else:  # str
            out[ss.name] = _clean_str(raw)
    # Carry through the timer's unit (optional, not a required slot).
    if intent == "set_timer" and slots.get("duration_unit"):
        out["duration_unit"] = _clean_str(slots["duration_unit"])
    return out, None, None


def decode(
    intent: str,
    slots: Mapping[str, Any] | None,
    confidence: float,
    *,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    dry_run: bool = True,
) -> ActionRequest | RejectResult:
    """Turn a raw model result into an :class:`ActionRequest` or :class:`RejectResult`.

    Order of checks: OOV -> confidence -> slot validation. The first failure
    wins and produces a :class:`RejectResult`; a fully-valid command yields an
    :class:`ActionRequest` ready for the orchestrator.
    """
    slots = dict(slots or {})

    if intent != "oov" and intent not in KNOWN_INTENTS:
        return _reject_oov(intent)
    if intent == "oov":
        return _reject_oov("oov")

    if confidence is None or confidence < threshold:
        return _reject_low_conf(float(confidence or 0.0), threshold)

    spec = INTENT_SPECS[intent]
    normalized, missing, invalid = _validate_slots(intent, slots)
    if missing is not None:
        return RejectResult(reason="missing_slot", detail=missing,
                            reply_text="I don't understand.",
                            intent=intent)
    if invalid is not None:
        return RejectResult(reason="invalid_slot", detail=invalid,
                            reply_text="Invalid value. Try again.",
                            intent=intent)

    reply = _render(spec["reply"], normalized, intent)
    return ActionRequest(
        intent=intent,
        category=spec["category"],
        slots=normalized,
        action_code=spec["code"],
        reply_text=reply,
        dry_run=dry_run,
        confidence=float(confidence),
    )
