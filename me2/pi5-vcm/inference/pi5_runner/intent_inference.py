"""NumPy frontend and ONNX Runtime wrapper for ME2 intent inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SAMPLE_RATE = 16_000
FRAME_LENGTH = 400
FRAME_HOP = 160
FFT_SIZE = FRAME_LENGTH
NUM_MEL_BINS = 80
NUM_INTENTS = 20
NUM_SLOT_HEADS = 6
MIN_LOG_ENERGY = 1e-10
INTENT_OUTPUT_NAME = "intent_probs"
SLOT_OUTPUT_NAMES = tuple(f"slot_{index}_probs" for index in range(NUM_SLOT_HEADS))
OUTPUT_NAMES = (INTENT_OUTPUT_NAME, *SLOT_OUTPUT_NAMES)
INTENT_NAMES = (
    "PLAY_MUSIC",
    "WEATHER",
    "TIME",
    "LIGHT_ON",
    "LIGHT_OFF",
    "PAUSE",
    "STOP",
    "NEXT",
    "VOLUME_UP",
    "VOLUME_DOWN",
    "CALL",
    "MESSAGE",
    "LIST_REMINDERS",
    "TIMER",
    "ALARM",
    "TEMPERATURE",
    "BRIGHTNESS",
    "COLOR",
    "CREATE_REMINDER",
    "OUT_OF_SCOPE",
)
SLOT_NAMES = (
    "TIMER",
    "ALARM",
    "TEMPERATURE",
    "BRIGHTNESS",
    "COLOR",
    "CREATE_REMINDER",
)


class _ModelInput(Protocol):
    name: str
    type: str


class _ModelOutput(Protocol):
    name: str


class _InferenceSession(Protocol):
    def get_inputs(self) -> list[_ModelInput]: ...

    def get_outputs(self) -> list[_ModelOutput]: ...

    def run(
        self, output_names: list[str], inputs: dict[str, np.ndarray]
    ) -> list[np.ndarray]: ...


@dataclass(frozen=True)
class IntentPrediction:
    intent_id: int
    intent_label: str
    confidence: float
    out_of_scope_confidence: float
    accepted: bool
    slot_ids: tuple[int, ...]


def _mel_filterbank() -> np.ndarray:
    frequencies = np.fft.rfftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE).astype(np.float32)
    mel_edges = np.linspace(
        np.float32(0.0),
        np.float32(2595.0 * np.log10(1.0 + (SAMPLE_RATE / 2) / 700.0)),
        NUM_MEL_BINS + 2,
        dtype=np.float32,
    )
    hz_edges = (700.0 * (np.power(10.0, mel_edges / 2595.0) - 1.0)).astype(
        np.float32
    )
    lower = hz_edges[:-2, None]
    center = hz_edges[1:-1, None]
    upper = hz_edges[2:, None]
    rising = (frequencies[None, :] - lower) / (center - lower)
    falling = (upper - frequencies[None, :]) / (upper - center)
    return np.maximum(np.minimum(rising, falling), 0.0).astype(np.float32)


_MEL_FILTERBANK = _mel_filterbank()
_HANN_WINDOW = np.hanning(FRAME_LENGTH + 1)[:-1].astype(np.float32)


def log_mel_features(pcm_samples: np.ndarray) -> np.ndarray:
    """Match the training frontend for one mono int16 waveform."""
    samples = np.asarray(pcm_samples)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("PCM audio must be a non-empty one-dimensional array")
    if samples.dtype != np.int16:
        raise ValueError("PCM audio must use signed 16-bit samples")

    waveform = samples.astype(np.float32) / np.float32(32768.0)
    left_padded = np.pad(waveform, (FRAME_LENGTH - 1, 0))
    frames = sliding_window_view(left_padded, FRAME_LENGTH)[::FRAME_HOP]
    spectrum = np.fft.rfft(frames * _HANN_WINDOW, n=FFT_SIZE, axis=1)
    power = (spectrum.real**2 + spectrum.imag**2).astype(np.float32)
    mel_power = power @ _MEL_FILTERBANK.T
    log_mel = np.log(np.maximum(mel_power, MIN_LOG_ENERGY)).astype(np.float32)
    return log_mel[None, :, :]


class OnnxIntentPredictor:
    """Run an exported ME2 probability model on mono 16-kHz PCM samples."""

    def __init__(
        self,
        model_path: str | Path,
        mel_stats_path: str | Path,
        *,
        session: _InferenceSession | None = None,
    ) -> None:
        model_file = Path(model_path)
        if session is None:
            if not model_file.is_file():
                raise FileNotFoundError(f"ME2 ONNX model does not exist: {model_file}")
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError(
                    "onnxruntime is required for ME2 inference; install it on the Pi"
                ) from exc
            session = ort.InferenceSession(
                str(model_file), providers=["CPUExecutionProvider"]
            )
        self.session = session
        self._validate_model_contract()
        self.mean, self.std = self._load_statistics(Path(mel_stats_path))

    def _validate_model_contract(self) -> None:
        inputs = {item.name: item for item in self.session.get_inputs()}
        if set(inputs) != {"mel_features", "frame_mask"}:
            raise ValueError(
                "ME2 model inputs must be mel_features and frame_mask; "
                f"found {sorted(inputs)}"
            )
        if inputs["mel_features"].type != "tensor(float)":
            raise ValueError("ME2 mel_features input must use float elements")
        if inputs["frame_mask"].type != "tensor(bool)":
            raise ValueError("ME2 frame_mask input must use bool elements")
        outputs = {item.name for item in self.session.get_outputs()}
        if outputs != set(OUTPUT_NAMES):
            raise ValueError(
                "ME2 model output contract must include intent and slot "
                f"probabilities; found {sorted(outputs)}"
            )

    @staticmethod
    def _load_statistics(path: Path) -> tuple[np.ndarray, np.ndarray]:
        if not path.is_file():
            raise FileNotFoundError(f"ME2 mel statistics do not exist: {path}")
        with np.load(path, allow_pickle=False) as statistics:
            if set(statistics.files) != {"mean", "std"}:
                raise ValueError(
                    "mel statistics file must contain only mean and std arrays"
                )
            mean = np.asarray(statistics["mean"], dtype=np.float32)
            std = np.asarray(statistics["std"], dtype=np.float32)
        if mean.shape != (NUM_MEL_BINS,) or std.shape != (NUM_MEL_BINS,):
            raise ValueError(
                f"mel statistics arrays must have shape ({NUM_MEL_BINS},)"
            )
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("mel statistics must be finite")
        if np.any(std <= 0):
            raise ValueError("mel normalization standard deviations must be positive")
        return mean, std

    def predict(
        self, pcm_samples: np.ndarray, *, threshold: float
    ) -> IntentPrediction:
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("intent threshold must be finite and in [0, 1]")
        features = log_mel_features(pcm_samples)
        features = ((features - self.mean) / self.std).astype(np.float32)
        frame_mask = np.ones(features.shape[:2], dtype=np.bool_)
        outputs = self.session.run(
            list(OUTPUT_NAMES),
            {"mel_features": features, "frame_mask": frame_mask},
        )
        if len(outputs) != len(OUTPUT_NAMES):
            raise RuntimeError(
                f"ME2 model returned {len(outputs)} outputs; "
                f"expected {len(OUTPUT_NAMES)}"
            )

        intent_probs = np.asarray(outputs[0], dtype=np.float32)
        slot_probs = np.stack(
            [np.asarray(output, dtype=np.float32) for output in outputs[1:]],
            axis=1,
        )
        if intent_probs.shape != (1, NUM_INTENTS):
            raise RuntimeError(
                f"unexpected ME2 intent output shape: {intent_probs.shape}"
            )
        if slot_probs.shape != (1, NUM_SLOT_HEADS, 3):
            raise RuntimeError(f"unexpected ME2 slot output shape: {slot_probs.shape}")
        if (
            not np.isfinite(intent_probs).all()
            or not np.isfinite(slot_probs).all()
            or np.any(intent_probs < 0)
            or np.any(intent_probs > 1)
            or np.any(slot_probs < 0)
            or np.any(slot_probs > 1)
        ):
            raise RuntimeError("ME2 model returned invalid probabilities")

        in_scope_probs = intent_probs[0, :-1]
        intent_id = int(in_scope_probs.argmax())
        confidence = float(in_scope_probs[intent_id])
        out_of_scope_confidence = float(intent_probs[0, -1])
        accepted = confidence >= threshold and confidence > out_of_scope_confidence
        return IntentPrediction(
            intent_id=intent_id,
            intent_label=INTENT_NAMES[intent_id],
            confidence=confidence,
            out_of_scope_confidence=out_of_scope_confidence,
            accepted=accepted,
            slot_ids=tuple(int(value) for value in slot_probs[0].argmax(axis=1)),
        )
