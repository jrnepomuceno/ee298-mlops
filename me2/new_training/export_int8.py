"""Export an intent checkpoint to FP32 ONNX, then dynamic INT8 ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from new_training.model import IntentModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fp32-output", default="new_training/runs/intent_v1/model_fp32.onnx")
    parser.add_argument("--int8-output", default="new_training/runs/intent_v1/model_int8.onnx")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("task") != "intent_classification_only":
        raise ValueError("checkpoint is not a new_training intent-only model")
    model = IntentModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    fp32_path = Path(args.fp32_output)
    int8_path = Path(args.int8_output)
    fp32_path.parent.mkdir(parents=True, exist_ok=True)
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros((1, 120, 80), dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        str(fp32_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["mels"],
        output_names=["intent_logits"],
        dynamic_axes={"mels": {0: "batch_size", 1: "time"},
                      "intent_logits": {0: "batch_size"}},
    )

    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx.checker.check_model(onnx.load(str(fp32_path)))
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    onnx.checker.check_model(onnx.load(str(int8_path)))
    size_mb = int8_path.stat().st_size / 1e6
    print(f"[onnx] fp32={fp32_path} ({fp32_path.stat().st_size / 1e6:.2f} MB)")
    print(f"[onnx] int8={int8_path} ({size_mb:.2f} MB; size is informational)")

    if args.verify:
        import onnxruntime as ort

        session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        for frames in (80, 240, 640):
            sample = np.random.default_rng(frames).normal(
                0.0, 1.0, size=(2, frames, 80)).astype(np.float32)
            with torch.no_grad():
                expected = model(torch.from_numpy(sample)).numpy()
            actual = session.run(["intent_logits"], {"mels": sample})[0]
            maximum_error = float(np.max(np.abs(expected - actual)))
            agreement = float(np.mean(expected.argmax(axis=1) == actual.argmax(axis=1)))
            print(f"[verify] frames={frames} max_abs_logit_error={maximum_error:.5f} "
                  f"argmax_agreement={agreement:.3f}")


if __name__ == "__main__":
    main()
