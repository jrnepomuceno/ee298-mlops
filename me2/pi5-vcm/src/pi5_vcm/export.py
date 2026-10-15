from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxconverter_common import float16
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from torch import Tensor, nn

from pi5_vcm.dataset import NUM_INTENTS, ParquetAudioDataset, create_dataloader
from pi5_vcm.features import LogMelFrontend
from pi5_vcm.model import ME2Model, NUM_SLOT_HEADS

INPUT_NAMES = ("mel_features", "frame_mask")
OUTPUT_NAMES = ("intent_probs",) + tuple(
    f"slot_{index}_probs" for index in range(NUM_SLOT_HEADS)
)
DEFAULT_CALIBRATION_SAMPLES = 128


class ProbabilityModel(nn.Module):
    def __init__(self, model: ME2Model) -> None:
        super().__init__()
        self.model = model

    def forward(self, mel_features: Tensor, frame_mask: Tensor) -> tuple[Tensor, ...]:
        logits = self.model(mel_features, frame_mask)
        intent_probs = torch.softmax(logits["intent_logits"], dim=-1)
        slot_probs = torch.softmax(logits["slot_logits"], dim=-1).unbind(dim=1)
        return (intent_probs, *slot_probs)


class _ManifestCalibrationReader(CalibrationDataReader):
    def __init__(
        self,
        manifest_path: Path,
        *,
        dataset_dir: Path | None,
        frontend: LogMelFrontend,
        sample_count: int,
    ) -> None:
        dataset = ParquetAudioDataset(manifest_path, dataset_dir=dataset_dir)
        self.records = dataset.records
        if sample_count < 1:
            raise ValueError("calibration sample count must be positive")
        if sample_count > len(self.records):
            raise ValueError(
                f"requested {sample_count} calibration clips but training manifest "
                f"contains only {len(self.records)}"
            )
        ranked = sorted(
            range(len(self.records)),
            key=lambda index: hashlib.sha256(
                self.records[index].clip_id.encode("utf-8")
            ).digest(),
        )
        self.indices = ranked[:sample_count]
        self.clip_ids = [self.records[index].clip_id for index in self.indices]
        self.dataset = dataset
        self.frontend = frontend.eval().cpu()
        self.position = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.position == len(self.indices):
            return None
        sample = self.dataset[self.indices[self.position]]
        self.position += 1
        audio = sample["audio"].unsqueeze(0)
        lengths = torch.tensor([audio.shape[1]], dtype=torch.long)
        with torch.inference_mode():
            features, frame_mask = self.frontend(audio, lengths)
        return {
            INPUT_NAMES[0]: features.numpy().astype(np.float32, copy=False),
            INPUT_NAMES[1]: frame_mask.numpy(),
        }

    def rewind(self) -> None:
        self.position = 0


def _checkpoint_model(checkpoint_path: Path) -> tuple[ME2Model, LogMelFrontend]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"invalid training checkpoint: {checkpoint_path}")
    model = ME2Model()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    mean = checkpoint.get("mel_mean")
    std = checkpoint.get("mel_std")
    if not isinstance(mean, Tensor) or not isinstance(std, Tensor):
        raise ValueError("checkpoint is missing its training-only mel statistics")
    frontend = LogMelFrontend(mean=mean, std=std).eval()
    return model, frontend


def _repair_fp16_cast_targets(model: onnx.ModelProto) -> None:
    value_info = (
        *model.graph.input,
        *model.graph.value_info,
        *model.graph.output,
    )
    element_types = {
        value.name: value.type.tensor_type.elem_type
        for value in value_info
        if value.type.HasField("tensor_type")
    }
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        if any(
            element_types.get(output_name) == onnx.TensorProto.FLOAT16
            for output_name in node.output
        ):
            for attribute in node.attribute:
                if attribute.name == "to":
                    attribute.i = onnx.TensorProto.FLOAT16


def export_candidates(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    calibration_reader: _ManifestCalibrationReader,
) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model, _ = _checkpoint_model(Path(checkpoint_path))
    export_model = ProbabilityModel(model).eval()
    mel_features = torch.zeros((1, 32, 80), dtype=torch.float32)
    frame_mask = torch.ones((1, 32), dtype=torch.bool)
    float32_path = destination / "model_fp32.onnx"
    torch.onnx.export(
        export_model,
        (mel_features, frame_mask),
        float32_path,
        input_names=list(INPUT_NAMES),
        output_names=list(OUTPUT_NAMES),
        dynamic_axes={
            INPUT_NAMES[0]: {0: "batch", 1: "frames"},
            INPUT_NAMES[1]: {0: "batch", 1: "frames"},
            **{name: {0: "batch"} for name in OUTPUT_NAMES},
        },
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(float32_path))

    float16_path = destination / "model_fp16.onnx"
    fp16_model = float16.convert_float_to_float16(
        onnx.load(float32_path), keep_io_types=True
    )
    _repair_fp16_cast_targets(fp16_model)
    fp16_model = onnx.shape_inference.infer_shapes(fp16_model)
    onnx.checker.check_model(fp16_model)
    onnx.save(fp16_model, float16_path)

    int8_path = destination / "model_int8.onnx"
    calibration_reader.rewind()
    quantize_static(
        model_input=str(float32_path),
        model_output=str(int8_path),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
        },
    )
    onnx.checker.check_model(onnx.load(int8_path))
    return {
        "fp32": str(float32_path),
        "fp16": str(float16_path),
        "int8": str(int8_path),
    }


def _probabilities_from_outputs(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    intent_probs = np.asarray(outputs[0], dtype=np.float32)
    slot_probs = np.stack(
        [np.asarray(output, dtype=np.float32) for output in outputs[1:]], axis=1
    )
    if intent_probs.ndim != 2 or intent_probs.shape[1] != NUM_INTENTS:
        raise ValueError(f"unexpected intent output shape: {intent_probs.shape}")
    if slot_probs.ndim != 3 or slot_probs.shape[1:] != (NUM_SLOT_HEADS, 3):
        raise ValueError(f"unexpected slot output shape: {slot_probs.shape}")
    if not np.isfinite(intent_probs).all() or not np.isfinite(slot_probs).all():
        raise ValueError("ONNX inference returned non-finite probabilities")
    return intent_probs, slot_probs


def _infer_manifest(
    *,
    model_path: Path | None,
    model: ME2Model,
    frontend: LogMelFrontend,
    manifest_path: Path,
    dataset_dir: Path | None,
    batch_size: int,
) -> dict[str, np.ndarray]:
    loader = create_dataloader(
        manifest_path,
        batch_size=batch_size,
        shuffle=False,
        dataset_dir=dataset_dir,
    )
    session = (
        ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        if model_path is not None
        else None
    )
    all_intents: list[np.ndarray] = []
    all_slots: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_slot_labels: list[np.ndarray] = []
    all_slot_masks: list[np.ndarray] = []
    for batch in loader:
        waveforms = batch["audio"]
        lengths = batch["audio_lengths"]
        with torch.inference_mode():
            features, frame_mask = frontend(waveforms, lengths)
            if session is None:
                output = ProbabilityModel(model)(features, frame_mask)
                outputs = [tensor.cpu().numpy() for tensor in output]
            else:
                outputs = session.run(
                    list(OUTPUT_NAMES),
                    {
                        INPUT_NAMES[0]: features.numpy().astype(np.float32, copy=False),
                        INPUT_NAMES[1]: frame_mask.numpy(),
                    },
                )
        intents, slots = _probabilities_from_outputs(outputs)
        all_intents.append(intents)
        all_slots.append(slots)
        all_labels.append(batch["intent_targets"].numpy())
        all_slot_labels.append(batch["slot_targets"].numpy())
        all_slot_masks.append(batch["slot_mask"].numpy())
    return {
        "intent_probs": np.concatenate(all_intents),
        "slot_probs": np.concatenate(all_slots),
        "intent_labels": np.concatenate(all_labels),
        "slot_labels": np.concatenate(all_slot_labels),
        "slot_masks": np.concatenate(all_slot_masks),
    }


def _tune_threshold(
    validation: dict[str, np.ndarray],
    tune_oos: dict[str, np.ndarray],
) -> dict[str, Any]:
    validation_labels = validation["intent_labels"]
    in_scope = validation_labels < NUM_INTENTS - 1
    if not in_scope.any():
        raise ValueError("validation manifest contains no in-scope examples")
    validation_probs = validation["intent_probs"]
    tune_probs = tune_oos["intent_probs"]
    if not tune_probs.size:
        raise ValueError("tune_oos manifest contains no examples")
    if np.any(tune_oos["intent_labels"] != NUM_INTENTS - 1):
        raise ValueError("tune_oos manifest must contain only OUT_OF_SCOPE examples")

    thresholds = np.arange(50, 100, dtype=np.int32) / 100
    validation_max = validation_probs[:, : NUM_INTENTS - 1].max(axis=1)
    validation_accept = validation_max[:, None] >= thresholds[None, :]
    validation_accept &= (
        validation_max[:, None] > validation_probs[:, NUM_INTENTS - 1, None]
    )
    tune_max = tune_probs[:, : NUM_INTENTS - 1].max(axis=1)
    tune_accept = tune_max[:, None] >= thresholds[None, :]
    tune_accept &= tune_max[:, None] > tune_probs[:, NUM_INTENTS - 1, None]

    candidates: list[dict[str, Any]] = []
    for index, threshold in enumerate(thresholds):
        recall = float(validation_accept[in_scope, index].mean())
        false_accepts = int(tune_accept[:, index].sum())
        candidates.append(
            {
                "tau": round(float(threshold), 2),
                "in_scope_recall": recall,
                "tune_oos_false_accepts": false_accepts,
                "tune_oos_false_accept_rate": false_accepts / len(tune_probs),
            }
        )
    meeting_constraint = [item for item in candidates if item["in_scope_recall"] >= 0.95]
    if meeting_constraint:
        selected = min(
            meeting_constraint,
            key=lambda item: (item["tune_oos_false_accepts"], item["tau"]),
        )
    else:
        selected = min(
            candidates,
            key=lambda item: (
                -item["in_scope_recall"],
                item["tune_oos_false_accepts"],
                item["tau"],
            ),
        )
    selected = {
        **selected,
        "constraint_met": selected["in_scope_recall"] >= 0.95,
        "validation_in_scope_examples": int(in_scope.sum()),
        "tune_oos_examples": int(len(tune_probs)),
    }
    return selected


def _candidate_metrics(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, Any]:
    baseline_intent = baseline["intent_probs"]
    candidate_intent = candidate["intent_probs"]
    labels = candidate["intent_labels"]
    slot_labels = candidate["slot_labels"]
    slot_masks = candidate["slot_masks"].astype(bool)
    baseline_slot = baseline["slot_probs"]
    candidate_slot = candidate["slot_probs"]
    per_head_accuracy: list[dict[str, Any]] = []
    for slot_index in range(NUM_SLOT_HEADS):
        valid = slot_masks[:, slot_index] & (slot_labels[:, slot_index] >= 0)
        if not valid.any():
            accuracy = None
            examples = 0
        else:
            accuracy = float(
                (
                    candidate_slot[valid, slot_index].argmax(axis=1)
                    == slot_labels[valid, slot_index]
                ).mean()
            )
            examples = int(valid.sum())
        base_accuracy = (
            float(
                (
                    baseline_slot[valid, slot_index].argmax(axis=1)
                    == slot_labels[valid, slot_index]
                ).mean()
            )
            if examples
            else None
        )
        per_head_accuracy.append(
            {
                "slot_index": slot_index,
                "examples": examples,
                "candidate_accuracy": accuracy,
                "fp32_accuracy": base_accuracy,
                "accuracy_drop": (
                    base_accuracy - accuracy
                    if accuracy is not None and base_accuracy is not None
                    else None
                ),
            }
        )
    agreement = float(
        (
            candidate_intent.argmax(axis=1) == baseline_intent.argmax(axis=1)
        ).mean()
    )
    probability_error = float(
        np.abs(candidate_intent - baseline_intent).mean()
    )
    return {
        "intent_argmax_agreement": agreement,
        "intent_mean_absolute_probability_error": probability_error,
        "slot_accuracy_by_head": per_head_accuracy,
        "candidate_intent_accuracy": float(
            (candidate_intent.argmax(axis=1) == labels).mean()
        ),
    }


def assess_candidates(
    *,
    model_paths: dict[str, str],
    checkpoint_path: str | Path,
    validation_manifest: str | Path,
    tune_oos_manifest: str | Path,
    dataset_dir: str | Path | None,
    report_path: str | Path,
    batch_size: int = 16,
) -> dict[str, Any]:
    model, frontend = _checkpoint_model(Path(checkpoint_path))
    frontend.eval()
    dataset_root = Path(dataset_dir) if dataset_dir is not None else None
    validation_args = {
        "model": model,
        "frontend": frontend,
        "manifest_path": Path(validation_manifest),
        "dataset_dir": dataset_root,
        "batch_size": batch_size,
    }
    baseline = _infer_manifest(model_path=None, **validation_args)
    fp32_validation = _infer_manifest(
        model_path=Path(model_paths["fp32"]), **validation_args
    )
    tune_oos_fp32 = _infer_manifest(
        model_path=Path(model_paths["fp32"]),
        model=model,
        frontend=frontend,
        manifest_path=Path(tune_oos_manifest),
        dataset_dir=dataset_root,
        batch_size=batch_size,
    )
    report: dict[str, Any] = {
        "validation_examples": int(len(baseline["intent_labels"])),
        "candidate_metrics": {},
        "models": {},
        "calibration": {},
    }
    baseline_fp32 = _candidate_metrics(baseline, fp32_validation)
    report["candidate_metrics"]["fp32"] = {
        **baseline_fp32,
        "tau": _tune_threshold(fp32_validation, tune_oos_fp32),
        "acceptance": "reference",
    }
    report["models"]["fp32"] = _model_file_metadata(Path(model_paths["fp32"]))

    for name in ("fp16", "int8"):
        model_path = Path(model_paths[name])
        try:
            candidate_validation = _infer_manifest(
                model_path=model_path, **validation_args
            )
            candidate_tune = _infer_manifest(
                model_path=model_path,
                model=model,
                frontend=frontend,
                manifest_path=Path(tune_oos_manifest),
                dataset_dir=dataset_root,
                batch_size=batch_size,
            )
            metrics = _candidate_metrics(fp32_validation, candidate_validation)
            tau = _tune_threshold(candidate_validation, candidate_tune)
            numerical_pass = (
                metrics["intent_argmax_agreement"] >= 0.99
                and metrics["intent_mean_absolute_probability_error"] <= 0.01
                and all(
                    item["accuracy_drop"] is None or item["accuracy_drop"] <= 0.01
                    for item in metrics["slot_accuracy_by_head"]
                )
            )
            tau_pass = (
                tau["constraint_met"]
                and tau["tune_oos_false_accept_rate"]
                <= report["candidate_metrics"]["fp32"]["tau"][
                    "tune_oos_false_accept_rate"
                ]
                + 0.01
            )
            report["candidate_metrics"][name] = {
                **metrics,
                "tau": tau,
                "numerical_gates_passed": numerical_pass,
                "tau_gate_passed": tau_pass,
                "acceptance": "passed" if numerical_pass and tau_pass else "failed",
            }
        except (RuntimeError, ValueError, onnx.checker.ValidationError) as exc:
            report["candidate_metrics"][name] = {
                "acceptance": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        report["models"][name] = _model_file_metadata(model_path)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _model_file_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    graph = onnx.load(path)
    operators: dict[str, int] = {}
    for node in graph.graph.node:
        operators[node.op_type] = operators.get(node.op_type, 0) + 1
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "operators": operators,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export and validate FP32, FP16, and static QDQ INT8 ME2 ONNX models."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--tune-oos-manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLES,
    )
    parser.add_argument("--batch-size", type=int, default=16)
    arguments = parser.parse_args()

    _, frontend = _checkpoint_model(arguments.checkpoint)
    calibration_reader = _ManifestCalibrationReader(
        arguments.train_manifest,
        dataset_dir=arguments.dataset_dir,
        frontend=frontend,
        sample_count=arguments.calibration_samples,
    )
    model_paths = export_candidates(
        arguments.checkpoint,
        arguments.output_dir,
        calibration_reader=calibration_reader,
    )
    report = assess_candidates(
        model_paths=model_paths,
        checkpoint_path=arguments.checkpoint,
        validation_manifest=arguments.validation_manifest,
        tune_oos_manifest=arguments.tune_oos_manifest,
        dataset_dir=arguments.dataset_dir,
        report_path=Path(arguments.output_dir) / "export_report.json",
        batch_size=arguments.batch_size,
    )
    report["calibration"] = {
        "method": "static QDQ MinMax",
        "source": "training manifest only",
        "sample_count": len(calibration_reader.clip_ids),
        "clip_ids": calibration_reader.clip_ids,
        "clip_ids_sha256": hashlib.sha256(
            "\n".join(calibration_reader.clip_ids).encode("utf-8")
        ).hexdigest(),
        "per_channel_weights": True,
        "activation_type": "QUInt8",
        "weight_type": "QInt8",
    }
    report_path = Path(arguments.output_dir) / "export_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
