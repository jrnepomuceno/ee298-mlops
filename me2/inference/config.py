"""Central configuration for the Pi5-VCM project.

Everything the other modules need in one place: audio feature settings,
the intent taxonomy, and the constrained CTC vocabulary used by the slot
head. Change the command set or the slot vocabulary here and the rest of
the code picks it up.
"""

from __future__ import annotations

# ------------------------------------------------------------------ audio --
SAMPLE_RATE = 16_000          # Hz, mono
N_MELS = 80                   # log-mel bins
FRAME_LENGTH_MS = 25.0        # analysis window
FRAME_SHIFT_MS = 10.0         # hop size

# ----------------------------------------------------------------- intents --
# 16 classes: 15 command intents + 1 out-of-vocabulary (OOV) class.
INTENTS = [
    "turn_on_lights",
    "turn_off_lights",
    "dim_lights",
    "set_temperature",
    "play_music",
    "pause_music",
    "stop_music",
    "set_timer",
    "set_alarm",
    "cancel_timer",
    "remind",
    "call",
    "what_time",
    "what_weather",
    "what_reminders",
    "oov",
]
INTENT_TO_ID = {name: i for i, name in enumerate(INTENTS)}
NUM_INTENTS = len(INTENTS)
OOV_INTENT = "oov"

# ------------------------------------------------------------- CTC vocab --
# The slot head is a CTC decoder over a *constrained* vocabulary: digits,
# clock words, and a small set of slot words (contacts, reminder keywords).
# No open language -- that is what keeps the head small and LLM-free.

CTC_BLANK = 0
CTC_UNK = 1
CTC_DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

CTC_WORDS = [
    # units
    "percent", "degrees", "minutes", "minute", "hours", "hour",
    "seconds", "second",
    # clock words
    "am", "pm", "half", "past", "quarter", "to", "o", "clock", "midnight",
    # function words that appear in commands
    "a", "an", "the", "and", "at", "in", "on", "for", "me", "my", "please",
    "set", "turn", "lights", "light", "temperature", "music", "timer",
    "alarm", "remind", "call", "time", "weather", "reminders", "pause",
    "stop", "cancel", "play", "dim", "what", "are", "is",
    # contact names
    "mom", "dad", "john", "jane", "maria", "carlos",
    # reminder keywords
    "water", "plants", "buy", "milk", "bread", "eggs", "meeting", "doctor",
    "gym", "homework",
]


def build_ctc_vocab() -> list[str]:
    """Token order: <blank>, <unk>, digits, words (duplicates removed)."""
    vocab = ["<blank>", "<unk>"] + list(CTC_DIGITS)
    for word in CTC_WORDS:
        if word not in vocab:
            vocab.append(word)
    return vocab


def transcript_to_tokens(transcript: str) -> list[str]:
    """Split a transcript into constrained-vocab tokens.

    Digits are tokenized one character at a time ("40" -> "4", "0");
    words outside the vocabulary map to <unk> downstream.
    """
    tokens: list[str] = []
    for raw in transcript.lower().split():
        word = raw.strip(".,!?;:'\"()")
        if not word:
            continue
        if word.isdigit():
            tokens.extend(list(word))
        else:
            tokens.append(word)
    return tokens


CTC_VOCAB = build_ctc_vocab()
CTC_TOKEN_TO_ID = {tok: i for i, tok in enumerate(CTC_VOCAB)}
CTC_ID_TO_TOKEN = {i: tok for tok, i in CTC_TOKEN_TO_ID.items()}
CTC_VOCAB_SIZE = len(CTC_VOCAB)

# -------------------------------------------------------------------- loss --
CE_WEIGHT = 1.0     # weight of the intent cross-entropy term
CTC_WEIGHT = 1.0    # weight of the CTC term
