#!/usr/bin/env python3
"""Quantize the exported ONNX VCM to int8 for Raspberry Pi 5 deployment.

The FP32 export is ~7.6 MB (4 bytes/param). Dynamic quantization reduces
eligible weight storage and typically speeds up CPU inference on the Pi.
The resulting file size is reported but is not subject to a fixed limit.

Why dynamic (no calibration set):
  * The model is small and the Pi runs CPU-only, so the extra accuracy of a
    static/calibrated int8 model is not worth needing a representative
    calibration corpus. Dynamic quantization is deterministic and fast.
  * It quantizes MatMul/Gemm weights to int8 and leaves activations in fp32,
    which is the right trade-off for a latency-bound command model.

Usage:
    python model/quantize_onnx.py \
        --input  optimized_model.onnx \
        --output vcm_model_int8.onnx \
        --verify

Notes:
  * Run this AFTER optimize_onnx.py (so the graph is already pruned/fused),
    or directly on the raw export -- both work.
  * Verify on the Pi too: the benchmark here is a desktop CPU proxy, not the
    Cortex-A76. Use the numbers as a sanity check, then re-time on-device.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Project root on path so `import config` / `from model import VCM` resolve
# regardless of where this is launched from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _quantize(input_path: str, output_path: str) -> None:
    """Dynamic int8 quantization via onnxruntime.quantization."""
    try:
        from onnxruntime.quantization import (
            QuantType,
            quantize_dynamic,
        )
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "onnxruntime with quantization support is required. "
            "Install: pip install onnxruntime"
        ) from exc

    print(f"[quant] input : {input_path} ({os.path.getsize(input_path)/1e6:.2f} MB)")
    print("[quant] scheme: dynamic, weights->int8, activations->fp32")
    quantize_dynamic(
        input_path,
        output_path,
        weight_type=QuantType.QInt8,
    )
    print(f"[quant] output: {output_path} ({os.path.getsize(output_path)/1e6:.2f} MB)")


def _bench(path: str, frames: int = 100, mels: int = 80, runs: int = 50) -> None:
    """Latency proxy on the *current* CPU (NOT the Pi). Informational only."""
    try:
        from onnxruntime import InferenceSession, SessionOptions
    except ImportError:  # pragma: no cover
        print("[bench] onnxruntime not available; skipping benchmark")
        return

    opts = SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    # (1, T, 80) fp32, matching the export signature.
    dummy = np.random.randn(1, frames, mels).astype(np.float32)

    # warm-up
    for _ in range(5):
        sess.run(None, {in_name: dummy})

    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {in_name: dummy})
    dt = (time.perf_counter() - t0) / runs * 1000.0

    # RTF against a 16 kHz, 10 ms-hop utterance of `frames` frames.
    audio_ms = frames * 10.0
    rtf = (dt / 1000.0) / (audio_ms / 1000.0)
    print(f"[bench] {Path(path).name}: {dt:.2f} ms/infer "
          f"({runs} runs, {frames} frames) | RTF={rtf:.3f}")
    print(f"[bench] NOTE: measured on {os.uname().machine if hasattr(os,'uname') else 'host'}, "
          f"not the Pi5. Re-benchmark on-device.")


def main() -> int:
    p = argparse.ArgumentParser(description="int8-quantize the VCM ONNX model")
    p.add_argument("--input", default="optimized_model.onnx",
                   help="input ONNX (FP32, optimized) path")
    p.add_argument("--output", default="vcm_model_int8.onnx",
                   help="output int8 ONNX path")
    p.add_argument("--verify", action="store_true",
                   help="run a CPU latency proxy benchmark after quantizing")
    args = p.parse_args()

    if not Path(args.input).exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 1

    _quantize(args.input, args.output)

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"[quant] output size: {size_mb:.2f} MB (informational)")

    if args.verify:
        _bench(args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
