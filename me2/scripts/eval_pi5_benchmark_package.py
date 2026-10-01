#!/usr/bin/env python3
"""Run benchmark on pi5_benchmark_package on Raspberry Pi 5.

Evaluates v6_int8, v6_fp32, v1_int8, and v1_fp32 against pi5_benchmark_package.
Measures latency (mean, p50, p95), RTF, peak RSS, canonical intent accuracy,
and runtime intent accuracy.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.features import kaldi_fbank, load_wav_mono
from inference.ort_infer import run_utterance
from rpi5.v1_adapter import adapt_v1_result
from rpi5.v6_adapter import adapt_v6_result

HOP_MS = 10.0


def peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024.0 * 1024.0) if sys.platform == "darwin" else ru / 1024.0


def evaluate_model(name: str, onnx_path: Path, contract: dict, adapter_fn,
                   samples: list[dict], pkg_dir: Path, threads: int = 2) -> dict:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    labels = contract["labels"]

    model_latencies = []
    feature_latencies = []
    total_latencies = []
    audio_durations = []

    canonical_correct = 0
    canonical_total = 0
    runtime_correct = 0
    runtime_total = len(samples)

    per_canonical = {}
    per_runtime = {}

    for item in samples:
        audio_file = pkg_dir / item["audio_path"]
        wav = load_wav_mono(str(audio_file))
        audio_dur = len(wav) / 16000.0
        audio_durations.append(audio_dur)

        # 1. Feature extraction timing
        t_feat0 = time.perf_counter()
        features = kaldi_fbank(wav, snip_edges=bool(contract.get("feature_config", {}).get("snip_edges", True)))
        feat_ms = (time.perf_counter() - t_feat0) * 1000.0
        feature_latencies.append(feat_ms)

        # 2. Model inference timing
        t_infer0 = time.perf_counter()
        res = run_utterance(sess, wav, 400, labels, [], diagnostic_contract=contract)
        infer_ms = (time.perf_counter() - t_infer0) * 1000.0
        model_latencies.append(infer_ms)

        total_latencies.append(feat_ms + infer_ms)

        raw_pred = res["intent"]
        adapted = adapter_fn(res, contract)
        runtime_pred = adapted["intent"]

        ref_canonical = item.get("canonical_intent")
        ref_native = item["native_label"]

        # Canonical evaluation (mapped subset)
        if ref_canonical:
            canonical_total += 1
            is_can_ok = (raw_pred == ref_canonical)
            if is_can_ok:
                canonical_correct += 1
            if ref_canonical not in per_canonical:
                per_canonical[ref_canonical] = {"total": 0, "correct": 0}
            per_canonical[ref_canonical]["total"] += 1
            if is_can_ok:
                per_canonical[ref_canonical]["correct"] += 1

        # Runtime evaluation (full set including unmapped)
        is_run_ok = (runtime_pred == ref_native)
        if is_run_ok:
            runtime_correct += 1
        if ref_native not in per_runtime:
            per_runtime[ref_native] = {"total": 0, "correct": 0}
        per_runtime[ref_native]["total"] += 1
        if is_run_ok:
            per_runtime[ref_native]["correct"] += 1

    total_audio_s = sum(audio_durations)
    total_model_s = sum(model_latencies) / 1000.0
    total_pipeline_s = sum(total_latencies) / 1000.0

    return {
        "model_name": name,
        "onnx_path": str(onnx_path),
        "model_size_mb": round(onnx_path.stat().st_size / (1024 * 1024), 2),
        "threads": threads,
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "latency_ms": {
            "model_mean": round(float(np.mean(model_latencies)), 2),
            "model_p50": round(float(np.percentile(model_latencies, 50)), 2),
            "model_p95": round(float(np.percentile(model_latencies, 95)), 2),
            "feature_mean": round(float(np.mean(feature_latencies)), 2),
            "feature_p95": round(float(np.percentile(feature_latencies, 95)), 2),
            "total_mean": round(float(np.mean(total_latencies)), 2),
            "total_p95": round(float(np.percentile(total_latencies, 95)), 2),
        },
        "rtf": {
            "model_rtf": round(total_model_s / total_audio_s, 4),
            "total_pipeline_rtf": round(total_pipeline_s / total_audio_s, 4),
        },
        "accuracy": {
            "canonical_intent": {
                "correct": canonical_correct,
                "total": canonical_total,
                "accuracy": round(canonical_correct / canonical_total, 4) if canonical_total else 0.0,
                "per_class": per_canonical,
            },
            "runtime_intent": {
                "correct": runtime_correct,
                "total": runtime_total,
                "accuracy": round(runtime_correct / runtime_total, 4),
                "per_class": per_runtime,
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkg-dir", default="pi5_benchmark_package")
    parser.add_argument("--output", default="benchmark-results/pi5-benchmark-package-results.json")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    pkg_dir = Path(args.pkg_dir)
    if not pkg_dir.exists():
        sys.exit(f"error: package dir {pkg_dir} does not exist")

    jsonl_file = pkg_dir / "benchmark.jsonl"
    if not jsonl_file.exists():
        sys.exit(f"error: benchmark.jsonl not found in {pkg_dir}")

    samples = [json.loads(line) for line in jsonl_file.read_text(encoding="utf-8").strip().splitlines() if line.strip()]

    contract_v6_file = ROOT / "models/onnx/v6_15m/contract.json"
    contract_v1_file = ROOT / "new_training/intent_v1_labels.json"

    contract_v6 = json.loads(contract_v6_file.read_text(encoding="utf-8"))
    contract_v1 = json.loads(contract_v1_file.read_text(encoding="utf-8"))

    models_to_test = [
        ("v6_int8", ROOT / "models/onnx/v6_15m/model_int8.onnx", contract_v6, adapt_v6_result),
        ("v6_fp32", ROOT / "models/onnx/v6_15m/model_fp32.onnx", contract_v6, adapt_v6_result),
        ("v1_int8", Path.home() / "MyProjects/intent_v1/model_int8.onnx", contract_v1, adapt_v1_result),
        ("v1_fp32", Path.home() / "MyProjects/intent_v1/model_fp32.onnx", contract_v1, adapt_v1_result),
    ]

    results = {}
    for name, path, contract, adapter in models_to_test:
        if not path.exists():
            print(f"Skipping {name}: {path} not found")
            continue
        print(f"Benchmarking {name} ({path.name})...")
        res = evaluate_model(name, path, contract, adapter, samples, pkg_dir, threads=args.threads)
        results[name] = res
        print(f"  Canonical accuracy: {res['accuracy']['canonical_intent']['accuracy']:.2%} ({res['accuracy']['canonical_intent']['correct']}/{res['accuracy']['canonical_intent']['total']})")
        print(f"  Runtime accuracy  : {res['accuracy']['runtime_intent']['accuracy']:.2%} ({res['accuracy']['runtime_intent']['correct']}/{res['accuracy']['runtime_intent']['total']})")
        print(f"  Model p95 latency : {res['latency_ms']['model_p95']} ms (RTF: {res['rtf']['model_rtf']})")
        print(f"  Total p95 latency : {res['latency_ms']['total_p95']} ms (Total RTF: {res['rtf']['total_pipeline_rtf']})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved full benchmark report to {out_path}")


if __name__ == "__main__":
    main()
