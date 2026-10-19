from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from inference.pi5_runner import intent_inference as inference
from pi5_vcm.features import LogMelFrontend


class FakeSession:
    def __init__(
        self,
        intent_probs: np.ndarray | None = None,
        slot_probs: list[np.ndarray] | None = None,
        *,
        output_names: tuple[str, ...] = inference.OUTPUT_NAMES,
    ) -> None:
        self.intent_probs = (
            np.asarray(
                [[0.7] + [0.0] * 18 + [0.2]],
                dtype=np.float32,
            )
            if intent_probs is None
            else intent_probs
        )
        self.slot_probs = (
            [np.asarray([[0.1, 0.8, 0.1]], dtype=np.float32) for _ in range(6)]
            if slot_probs is None
            else slot_probs
        )
        self.inputs = [
            SimpleNamespace(name="mel_features", type="tensor(float)"),
            SimpleNamespace(name="frame_mask", type="tensor(bool)"),
        ]
        self.outputs = [SimpleNamespace(name=name) for name in output_names]
        self.run_inputs: dict[str, np.ndarray] | None = None

    def get_inputs(self) -> list[SimpleNamespace]:
        return self.inputs

    def get_outputs(self) -> list[SimpleNamespace]:
        return self.outputs

    def run(
        self, output_names: list[str], inputs: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        assert output_names == list(inference.OUTPUT_NAMES)
        self.run_inputs = inputs
        return [self.intent_probs, *self.slot_probs]


def write_stats(path: Path, mean: np.ndarray, std: np.ndarray) -> None:
    np.savez(path, mean=mean, std=std)


def test_numpy_frontend_matches_training_frontend() -> None:
    random = np.random.default_rng(42)
    pcm = random.integers(-10_000, 10_001, size=2_560, dtype=np.int16)
    mean = np.linspace(-2.0, 2.0, inference.NUM_MEL_BINS, dtype=np.float32)
    std = np.linspace(0.5, 1.5, inference.NUM_MEL_BINS, dtype=np.float32)

    actual = inference.log_mel_features(pcm)
    actual = (actual - mean) / std
    training_frontend = LogMelFrontend(
        mean=torch.from_numpy(mean),
        std=torch.from_numpy(std),
    ).eval()
    with torch.inference_mode():
        expected, frame_mask = training_frontend(
            torch.from_numpy(pcm.astype(np.float32) / 32768.0).unsqueeze(0),
            torch.tensor([len(pcm)]),
        )

    assert frame_mask.all()
    np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-5, atol=2e-5)


def test_predictor_normalizes_audio_and_returns_intent_and_slots(
    tmp_path: Path,
) -> None:
    stats_path = tmp_path / "mel_stats.npz"
    mean = np.linspace(-1.0, 1.0, inference.NUM_MEL_BINS, dtype=np.float32)
    std = np.linspace(0.5, 2.0, inference.NUM_MEL_BINS, dtype=np.float32)
    write_stats(stats_path, mean, std)
    session = FakeSession()
    predictor = inference.OnnxIntentPredictor(
        tmp_path / "unused.onnx", stats_path, session=session
    )
    pcm = np.arange(1_600, dtype=np.int16)

    result = predictor.predict(pcm, threshold=0.5)

    assert result.intent_id == 0
    assert result.intent_label == "PLAY_MUSIC"
    assert result.confidence == pytest.approx(0.7)
    assert result.out_of_scope_confidence == pytest.approx(0.2)
    assert result.accepted
    assert result.slot_ids == (1,) * inference.NUM_SLOT_HEADS
    assert session.run_inputs is not None
    assert session.run_inputs["mel_features"].dtype == np.float32
    assert session.run_inputs["mel_features"].shape == (1, 10, 80)
    assert session.run_inputs["frame_mask"].dtype == np.bool_
    np.testing.assert_array_equal(
        session.run_inputs["frame_mask"], np.ones((1, 10), dtype=np.bool_)
    )


@pytest.mark.parametrize(
    ("intent_probs", "threshold", "expected_accepted"),
    [
        ([[0.49] + [0.0] * 18 + [0.2]], 0.5, False),
        ([[0.7] + [0.0] * 18 + [0.7]], 0.5, False),
        ([[0.6] + [0.0] * 18 + [0.4]], 0.5, True),
    ],
)
def test_predictor_applies_confidence_and_oos_gate(
    tmp_path: Path,
    intent_probs: list[list[float]],
    threshold: float,
    expected_accepted: bool,
) -> None:
    stats_path = tmp_path / "mel_stats.npz"
    write_stats(
        stats_path,
        np.zeros(inference.NUM_MEL_BINS, dtype=np.float32),
        np.ones(inference.NUM_MEL_BINS, dtype=np.float32),
    )
    session = FakeSession(intent_probs=np.asarray(intent_probs, dtype=np.float32))
    predictor = inference.OnnxIntentPredictor(
        tmp_path / "unused.onnx", stats_path, session=session
    )

    result = predictor.predict(np.zeros(1_600, dtype=np.int16), threshold=threshold)

    assert result.accepted is expected_accepted


def test_predictor_rejects_model_contract_mismatch(tmp_path: Path) -> None:
    stats_path = tmp_path / "mel_stats.npz"
    write_stats(
        stats_path,
        np.zeros(inference.NUM_MEL_BINS, dtype=np.float32),
        np.ones(inference.NUM_MEL_BINS, dtype=np.float32),
    )
    session = FakeSession(output_names=("intent_probs",))

    with pytest.raises(ValueError, match="output contract"):
        inference.OnnxIntentPredictor(
            tmp_path / "unused.onnx", stats_path, session=session
        )


def test_predictor_rejects_wrong_mel_input_type(tmp_path: Path) -> None:
    stats_path = tmp_path / "mel_stats.npz"
    write_stats(
        stats_path,
        np.zeros(inference.NUM_MEL_BINS, dtype=np.float32),
        np.ones(inference.NUM_MEL_BINS, dtype=np.float32),
    )
    session = FakeSession()
    session.inputs[0].type = "tensor(double)"

    with pytest.raises(ValueError, match="mel_features input"):
        inference.OnnxIntentPredictor(
            tmp_path / "unused.onnx", stats_path, session=session
        )


@pytest.mark.parametrize(
    ("mean", "std"),
    [
        (np.zeros(79, dtype=np.float32), np.ones(79, dtype=np.float32)),
        (np.zeros(80, dtype=np.float32), np.zeros(80, dtype=np.float32)),
        (np.zeros(80, dtype=np.float32), np.full(80, np.nan, dtype=np.float32)),
    ],
)
def test_predictor_rejects_invalid_normalization_stats(
    tmp_path: Path,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    stats_path = tmp_path / "mel_stats.npz"
    write_stats(stats_path, mean, std)
    session = FakeSession()

    with pytest.raises(ValueError, match="mel statistics|standard deviations"):
        inference.OnnxIntentPredictor(
            tmp_path / "unused.onnx", stats_path, session=session
        )


def test_predictor_rejects_bad_pcm_and_threshold(tmp_path: Path) -> None:
    stats_path = tmp_path / "mel_stats.npz"
    write_stats(
        stats_path,
        np.zeros(inference.NUM_MEL_BINS, dtype=np.float32),
        np.ones(inference.NUM_MEL_BINS, dtype=np.float32),
    )
    predictor = inference.OnnxIntentPredictor(
        tmp_path / "unused.onnx", stats_path, session=FakeSession()
    )

    with pytest.raises(ValueError, match="signed 16-bit"):
        predictor.predict(np.zeros(1_600, dtype=np.float32), threshold=0.5)
    with pytest.raises(ValueError, match="threshold"):
        predictor.predict(np.zeros(1_600, dtype=np.int16), threshold=float("nan"))
