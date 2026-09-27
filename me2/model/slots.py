"""Rule-based slot parsing (pure-python, no torch).

Split out of ``model.model`` so the on-device ONNX inference path can import
``parse_slots`` without pulling in torch (the Pi's venv has no torch). The
function is identical to the one that lived in ``model.model``; it is
re-exported from there so ``from model import parse_slots`` keeps working.
"""
from __future__ import annotations

import re


def parse_slots(intent: str, transcript: str) -> dict:
    """Rule-based slot parser: (intent, transcript) -> slot dict.

    This is the non-LLM replacement for "the LLM extracts the slots".
    It only ever sees the constrained-vocab transcript, so regexes are
    small and deterministic.
    """
    import re

    slots: dict = {}
    t = transcript.lower()
    # The CTC vocab has one token per digit, so "18" decodes as "1 8".
    # Re-join space-separated digit tokens before the (\d+) regexes run,
    # otherwise multi-digit values truncate to their first digit (18 -> 8).
    t = re.sub(r"(?<=\d) (?=\d)", "", t)

    m = re.search(r"(\d+)\s*percent", t)
    if m:
        slots["percent"] = int(m.group(1))
    m = re.search(r"(\d+)\s*degrees?", t)
    if m:
        slots["temperature"] = int(m.group(1))
    m = re.search(r"(\d+)\s*(minutes?|seconds?)", t)
    if m:
        slots["duration"] = int(m.group(1))
        slots["duration_unit"] = m.group(2).rstrip("s")
    m = re.search(r"(\d+)\s*(am|pm)", t)
    if m:
        slots["time"] = f"{m.group(1)}:00 {m.group(2)}"
    if intent == "call":
        m = re.search(r"call\s+(\w+)", t)
        if m:
            slots["contact"] = m.group(1)
    if intent == "remind":
        m = re.search(r"remind\s+me\s+(?:to|about)\s+(.+)", t)
        if m:
            slots["note"] = m.group(1).strip()
    if intent == "set_alarm" and "time" not in slots:
        m = re.search(r"(\d+)", t)
        if m:
            slots["time"] = f"{m.group(1)}:00"
    return slots
