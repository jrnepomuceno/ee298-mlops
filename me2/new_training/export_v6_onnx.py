"""Export the v6 bounded-slot checkpoint as diagnostic INT8 ONNX."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .intent_v6_model import IntentBoundedSlots


SLOT_LABELS = (
    "timer_minute", "timer_second", "alarm_hour", "alarm_minute",
    "alarm_meridiem", "degrees", "percent", "task",
)
SLOT_OWNERS = (
    "TIMER", "TIMER", "ALARM", "ALARM", "ALARM", "TEMPERATURE",
    "BRIGHTNESS", "CREATE_REMINDER",
)
OUTPUT_NAMES = ("intent_logits", *(f"{name}_logits" for name in SLOT_LABELS))


def _slot_values(task_vocab: dict[str, int]) -> dict[str, list]:
    return {
        "timer_minute": list(range(61)),
        "timer_second": list(range(60)),
        "alarm_hour": list(range(1, 13)),
        "alarm_minute": [0, 15, 30, 45],
        "alarm_meridiem": ["AM", "PM"],
        "degrees": list(range(16, 41)),
        "percent": list(range(101)),
        "task": [name for name, _ in sorted(task_vocab.items(), key=lambda item: item[1])],
    }


def export(checkpoint_path: Path, labels_path: Path, output_dir: Path) -> tuple[Path, Path]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("task") != "intent_bounded_numeric_slots":
        raise ValueError("checkpoint is not an intent_bounded_numeric_slots model")
    if tuple(checkpoint.get("slot_labels", ())) != SLOT_LABELS:
        raise ValueError("checkpoint slot label order does not match the v6 exporter")

    label_contract = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = label_contract.get("labels") if isinstance(label_contract, dict) else label_contract
    if not isinstance(labels, list) or len(labels) != checkpoint["model_config"]["num_intents"]:
        raise ValueError("intent label count does not match v6 model configuration")

    task_vocab = checkpoint.get("task_vocab")
    if not isinstance(task_vocab, dict) or sorted(task_vocab.values()) != list(range(len(task_vocab))):
        raise ValueError("checkpoint task vocabulary must have contiguous integer IDs")
    state = checkpoint["model_state_dict"]
    slot_sizes = [int(state[f"heads.{index}.weight"].shape[0])
                  for index in range(len(SLOT_LABELS))]
    model = IntentBoundedSlots(checkpoint["model_config"], slot_sizes)
    model.load_state_dict(state, strict=True)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / "model_fp32.onnx"
    int8_path = output_dir / "model_int8.onnx"
    example_mels = torch.zeros((1, 120, 80), dtype=torch.float32)
    example_lengths = torch.tensor([120], dtype=torch.long)
    dynamic_axes = {
        "mels": {0: "batch_size", 1: "time"},
        "lengths": {0: "batch_size"},
    }
    dynamic_axes.update({name: {0: "batch_size"} for name in OUTPUT_NAMES})
    torch.onnx.export(
        model,
        (example_mels, example_lengths),
        str(fp32_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["mels", "lengths"],
        output_names=list(OUTPUT_NAMES),
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx.checker.check_model(onnx.load(str(fp32_path)))
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    onnx.checker.check_model(onnx.load(str(int8_path)))

    contract = {
        "task": checkpoint["task"],
        "labels": labels,
        "slot_labels": list(SLOT_LABELS),
        "slot_owners": dict(zip(SLOT_LABELS, SLOT_OWNERS)),
        "slot_values": _slot_values(task_vocab),
        "feature_config": {
            "sample_rate": 16000,
            "n_mels": 80,
            "frame_length_ms": 25.0,
            "frame_shift_ms": 10.0,
            "snip_edges": False,
            "max_frames": None,
        },
        "outputs": list(OUTPUT_NAMES),
        "source_checkpoint": checkpoint.get("source_checkpoint"),
    }
    contract_path = output_dir / "contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    # Validate the full INT8 output contract on a representative dynamic batch.
    session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    sample = np.random.default_rng(615).normal(0.0, 1.0, (2, 160, 80)).astype(np.float32)
    lengths = np.asarray([160, 137], dtype=np.int64)
    with torch.no_grad():
        expected = model(torch.from_numpy(sample), torch.from_numpy(lengths))
    actual = session.run(None, {"mels": sample, "lengths": lengths})
    if len(actual) != 1 + len(SLOT_LABELS):
        raise RuntimeError(f"unexpected ONNX output count: {len(actual)}")
    argmax_agreement = [
        float(np.mean(torch_output.numpy().argmax(axis=1) == onnx_output.argmax(axis=1)))
        for torch_output, onnx_output in zip(expected, actual)
    ]
    print("FP32:", fp32_path, fp32_path.stat().st_size, "bytes")
    print("INT8:", int8_path, int8_path.stat().st_size, "bytes")
    print("CONTRACT:", contract_path)
    print("ARGMAX_AGREEMENT:", argmax_agreement)
    return int8_path, contract_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--labels", type=Path,
                        default=Path(__file__).with_name("intent_v1_labels.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent /
                        "models/onnx/v6_15m")
    args = parser.parse_args()
    export(args.checkpoint, args.labels, args.output_dir)


if __name__ == "__main__":
    main()