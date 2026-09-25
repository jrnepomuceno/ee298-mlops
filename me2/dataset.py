"""Dataset for VCM.

Two ways to get samples:

1. **Real data** -- wav files plus a manifest (CSV or JSONL) with columns
   ``path, intent, transcript, speaker``.
2. **Synthetic data** -- generated on the fly from template commands
   (see :func:`build_synthetic_manifest`). This lets you develop and
   smoke-test the whole pipeline before real recordings exist. The
   synthesis is toy speech (formant tones), NOT real audio.
"""

from __future__ import annotations

import csv
import json
import random
import re
import warnings
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import config
from utils import audio_utils

# ------------------------------------------------------- template commands --
# Each template: (intent, list of word-variant templates, transcript builder).
# {x} placeholders are filled per sample; the transcript is what the CTC head
# must decode, so it is built from the SAME filled-in words.

TEMPLATES: list[tuple[str, list[str]]] = [
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
    ("cancel_timer", ["cancel the timer", "cancel timer",
                      "stop the timer"]),
    ("remind", ["remind me to water the plants",
                "remind me to buy milk",
                "remind me to buy bread",
                "remind me to buy eggs",
                "remind me about the meeting",
                "remind me about the doctor",
                "remind me about the gym",
                "remind me about homework"]),
    ("call", ["call {contact}", "call {contact} please"]),
    ("what_time", ["what time is it", "what is the time"]),
    ("what_weather", ["what is the weather", "what is the weather like",
                      "how is the weather"]),
    ("what_reminders", ["what are my reminders", "show my reminders"]),
]

# Out-of-vocabulary utterances: the model should label these "oov".
OOV_TEMPLATES: list[str] = [
    "tell me a joke", "who are you", "sing a song",
    "what is the meaning of life", "read the news",
    "how are you doing today", "tell me a story",
    "what is two plus two", "open the window",
    "close the door", "i am hungry", "good morning",
]

CONTACTS = ["mom", "dad", "john", "jane", "maria", "carlos"]


def _fill_template(template: str, rng: random.Random) -> str:
    def repl(match: re.Match) -> str:
        key = match.group(1)
        if key == "pct":
            return str(rng.choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
        if key == "num":
            return str(rng.choice([1, 2, 3, 5, 10, 15, 20, 30, 45, 60]))
        if key == "ampm":
            return rng.choice(["am", "pm"])
        if key == "contact":
            return rng.choice(CONTACTS)
        return match.group(0)

    return re.sub(r"\{(\w+)\}", repl, template)


def build_synthetic_manifest(n_samples: int, seed: int = 0,
                             oov_ratio: float = 0.15,
                             speakers: int = 8) -> list[dict]:
    """Create n_samples synthetic samples: (intent, words, transcript)."""
    rng = random.Random(seed)
    samples: list[dict] = []
    for i in range(n_samples):
        if rng.random() < oov_ratio:
            intent = config.OOV_INTENT
            words = _fill_template(rng.choice(OOV_TEMPLATES), rng).split()
        else:
            intent, templates = rng.choice(TEMPLATES)
            words = _fill_template(rng.choice(templates), rng).split()
        transcript = " ".join(words)
        samples.append({
            "intent": intent,
            "transcript": transcript,
            "words": words,
            "speaker": f"spk{i % speakers:02d}",
            "seed": rng.randint(0, 2**31 - 1),
        })
    return samples


def load_manifest(path) -> list[dict]:
    """Load a real-data manifest (CSV with header, or JSONL)."""
    path = Path(path)
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))



def manifest_fingerprint(samples: list[dict]) -> str:
    """Hash manifest content while ignoring machine-specific audio paths."""
    rows = []
    for sample in samples:
        fields = {key: value for key, value in sample.items() if key != "path"}
        rows.append(json.dumps(fields, sort_keys=True, separators=(",", ":")))
    payload = "\n".join(sorted(rows)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class VCMDataset(Dataset):
    """Yields (mel, intent_id, transcript) for one utterance.

    ``samples`` entries:
      synthetic: {"intent", "words", "speaker", "seed"}
      real:      {"path", "intent", "transcript"}
    """

    def __init__(self, samples: list[dict],
                 max_frames: int | None = None,
                 augment: bool = False,
                 mels_dir: str | Path | None = None,
                 split: str | None = None,
                 manifest_seed: int = 42,
                 mels_fingerprint: str | None = None,
                 noise_snr: float | None = None):
        self.samples = samples
        self.max_frames = max_frames
        self.augment = augment
        # When set (with augment=True), add white Gaussian noise to the
        # log-mel at a random SNR in [noise_snr, noise_snr + 10] dB.
        # Mel-domain so it works on the precomputed-mels path too.
        self.noise_snr = noise_snr
        self._cache: dict[tuple, torch.Tensor] = {}
        self._mels = None
        self._mels_path: Path | None = None
        # True (pre-padding) mel-frame count per row. Without it the
        # length-aware pool / CTC input_lengths degrade to the padded width.
        self._mels_lengths: np.ndarray | None = None
        if mels_dir is not None:
            if split not in {"train", "val", "test"}:
                raise ValueError("split must be train, val, or test with mels_dir")
            if any("path" not in sample for sample in samples):
                raise ValueError("precomputed mels require real samples with paths")
            mels_path = Path(mels_dir) / f"mels_{split}.npy"
            sidecar_path = mels_path.with_suffix(".json")
            if not mels_path.exists() or not sidecar_path.exists():
                raise FileNotFoundError(f"missing precomputed mel files for {split}")
            with open(sidecar_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            expected_fingerprint = mels_fingerprint or manifest_fingerprint(samples)
            if (metadata.get("seed") != manifest_seed or
                    metadata.get("manifest_fingerprint") != expected_fingerprint or
                    metadata.get("n_rows") != len(samples)):
                raise ValueError(f"precomputed mels metadata mismatch for {split}")
            shape = np.load(mels_path, mmap_mode="r", allow_pickle=False).shape
            if shape[0] != len(samples):
                raise ValueError(f"precomputed mel row mismatch for {split}")
            self._mels_path = mels_path
            lengths_path = mels_path.with_name(f"mels_{split}_lengths.npy")
            if lengths_path.exists():
                lengths = np.load(lengths_path, allow_pickle=False)
                if lengths.shape != (len(samples),):
                    raise ValueError(f"mel lengths row mismatch for {split}")
                self._mels_lengths = lengths.astype(np.int64)

    def __len__(self) -> int:
        return len(self.samples)

    def _augment_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Train-time augmentation: SpecAugment (+ optional SNR noise)."""
        mel = audio_utils.spec_augment(mel)
        if self.noise_snr is not None:
            snr = float(self.noise_snr + random.random() * 10.0)
            mel = audio_utils.spec_noise(mel, snr,
                                         np.random.default_rng())
        return mel

    def _mel(self, sample: dict, idx: int | None = None) -> torch.Tensor:
        """Raw (un-augmented, un-padded) mel for this sample."""
        if self._mels is not None:
            if idx is None:
                raise ValueError("precomputed mel lookup requires an index")
            return torch.from_numpy(np.asarray(self._mels[idx]).copy()).float()
        if self._mels_path is not None:
            # Open inside each DataLoader worker; Windows spawn cannot pickle
            # an mmap handle created in the parent process.
            self._mels = np.load(self._mels_path, mmap_mode="r", allow_pickle=False)
            return torch.from_numpy(np.asarray(self._mels[idx]).copy()).float()
        if "path" in sample:
            wav = audio_utils.load_wav_mono(sample["path"])
            return audio_utils.mel_spectrogram(wav)
        key = (sample["speaker"], tuple(sample["words"]), sample["seed"])
        if key in self._cache:
            return self._cache[key]
        wav = audio_utils.synthesize_utterance(
            sample["words"], sample["speaker"], seed=sample["seed"])
        mel = audio_utils.mel_spectrogram(torch.from_numpy(wav))
        if len(self._cache) < 2000:
            self._cache[key] = mel
        return mel

    def _true_length(self, sample: dict, idx: int, mel: torch.Tensor) -> int:
        """True mel-frame count (pre-padding), clamped to max_frames."""
        if self._mels_lengths is not None:
            length = int(self._mels_lengths[idx])
        else:
            length = mel.shape[0]
        if self.max_frames is not None:
            length = min(length, self.max_frames)
        return max(length, 1)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        mel = self._mel(sample, idx)
        length = self._true_length(sample, idx, mel)
        if self.augment:
            mel = self._augment_mel(mel)
        if self.max_frames is not None:
            mel = audio_utils.pad_or_trim(mel, self.max_frames)
        intent_id = config.INTENT_TO_ID.get(sample["intent"],
                                            config.INTENT_TO_ID[config.OOV_INTENT])
        transcript = sample.get("transcript") or " ".join(sample["words"])
        return mel, intent_id, transcript, length


def split_samples(samples: list[dict],
                  train_frac: float = 0.8,
                  val_frac: float = 0.1,
                  seed: int = 0
                  ) -> tuple[list[dict], list[dict], list[dict]]:
    """Split samples into (train, val, test).

    If every sample carries a non-empty ``split`` field with a value in
    ``{train, val, test}`` (the real dataset's manifest provides one),
    those labels are respected verbatim -- no reshuffling across splits,
    so the dataset's intended partition (speaker/phrase disjointness) is
    preserved and there is no cross-split leakage.

    Otherwise (e.g. synthetic manifests, which have no split column) this
    falls back to the legacy behaviour: shuffle once, slice by fraction.
    A warning is emitted in that case, since a random split can place
    same-speaker / same-phrase variants on both sides of the boundary.
    """
    splits = {str(s.get("split", "")).strip().lower() for s in samples}
    if splits and splits <= {"train", "val", "test"}:
        out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        for s in samples:
            out[str(s["split"]).strip().lower()].append(s)
        rng = random.Random(seed)
        for key in out:  # shuffle within each split only
            rng.shuffle(out[key])
        empty = [k for k, v in out.items() if not v]
        if empty:
            warnings.warn(
                f"split_samples: manifest 'split' column present but "
                f"split(s) {empty} contain no samples",
                stacklevel=2,
            )
        return out["train"], out["val"], out["test"]

    if samples:
        warnings.warn(
            "split_samples: no usable 'split' column in manifest; "
            "falling back to a random 80/10/10 split (leakage risk)",
            stacklevel=2,
        )
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_frac)
    n_val = int(len(shuffled) * val_frac)
    return (shuffled[:n_train],
            shuffled[n_train:n_train + n_val],
            shuffled[n_train + n_val:])
