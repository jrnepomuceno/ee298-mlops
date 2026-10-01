#!/usr/bin/env python3
"""Area 3 benchmark: VCM ONNX latency / RTF on the target device.

Measures the *model* (ONNX Runtime, CPU) at true utterance length -- no
pad-to-400 -- and reports per-utterance latency percentiles, RTF, peak RSS,
and the ARM clock before/after (thermal/throttle awareness for the 5V@3A Pi).

Design goals
------------
* Runs identically on the Mac (sanity) and the Raspberry Pi 5 (the number
  that counts). Depends only on onnxruntime + numpy (no torch), so the Pi
  needs no heavy install.
* Two input modes:
    synthetic (default)  deterministic random-mel sweeps across durations
                         -> pure latency/RTF, reproducible, no dataset needed
    --wav-dir DIR        real held-out clips -> latency AND quality
                         (intent accuracy, slot F1, OOV rejection) when
                         labels are present (labels.csv: path,intent[,transcript])
* Timing scope is ONLY session.run() (isolates the model). Mel extraction is
  timed separately so the full wake->intent path is also visible.

Pass bars (docs/plan.md): RTF <= 0.3 and latency p95 < 100 ms for a 1-2 s command.
Model size is reported for visibility but is not currently a pass/fail gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
    from onnxruntime import InferenceSession, SessionOptions
    from onnxruntime.capi.onnxruntime_pybind11_state import GraphOptimizationLevel
except ImportError as exc:  # pragma: no cover
    raise SystemExit("onnxruntime is required: pip install onnxruntime") from exc

# ---- project config (single source of truth) -----------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from model.onnx_deploy import (DEFAULT_ONNX_DIR, resolve_latest_onnx)  # noqa: E402
from model.slots import parse_slots  # noqa: E402
from inference.features import kaldi_fbank, load_wav_mono  # noqa: E402

SR = config.SAMPLE_RATE
N_MELS = config.N_MELS
HOP_MS = config.FRAME_SHIFT_MS          # 10 ms
INTENTS = config.INTENTS
OOV = "oov"
CTC_VOCAB = config.CTC_VOCAB
CTC_BLANK = config.CTC_BLANK


def ctc_greedy_decode(ctc_logits: np.ndarray) -> list[str]:
    """ctc_logits: (T, V) -> tokens (argmax, collapse repeats, drop blank)."""
    ids = ctc_logits.argmax(axis=-1)
    tokens: list[str] = []
    prev = -1
    for i in ids.tolist():
        if i != prev and i != CTC_BLANK:
            tokens.append(CTC_VOCAB[i])
        prev = i
    return tokens


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def frames_for_seconds(seconds: float) -> int:
    return max(4, int(round(seconds * 1000.0 / HOP_MS)))


def _vcgencmd(cmd: str) -> str:
    try:
        return subprocess.run(["vcgencmd", cmd], capture_output=True,
                              text=True, timeout=3).stdout.strip()
    except Exception:
        return ""


def env_snapshot() -> dict:
    snap = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "onnxruntime": ort.__version__,
    }
    clk = _vcgencmd("measure_clock arm")
    tmp = _vcgencmd("measure_temp")
    thr = _vcgencmd("get_throttled")
    if clk and "=" in clk:
        snap["arm_clock_ghz"] = round(int(clk.split("=")[1]) / 1e9, 3)
    if tmp:
        snap["temp_c"] = tmp
    if thr:
        snap["throttled"] = thr
    return snap


def peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024.0 * 1024.0) if sys.platform == "darwin" else ru / 1024.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_inputs(sess, in_name: str, mels: np.ndarray) -> dict:
    inputs = {in_name: mels}
    get_inputs = getattr(sess, "get_inputs", None)
    if callable(get_inputs) and any(item.name == "lengths" for item in get_inputs()):
        inputs["lengths"] = np.asarray([mels.shape[1]], dtype=np.int64)
    return inputs


def bench_one(sess, in_name, mels, n_runs, warmup):
    """Time session.run() only. Returns (mean_ms, p50_ms, p95_ms)."""
    inputs = _make_inputs(sess, in_name, mels)
    for _ in range(max(0, warmup)):
        sess.run(None, inputs)
    lats = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, inputs)
        lats.append((time.perf_counter() - t0) * 1000.0)
    lats = np.array(lats)
    return float(lats.mean()), float(np.percentile(lats, 50)), \
        float(np.percentile(lats, 95))


def run_synthetic(sess, in_name, args):
    rng = np.random.default_rng(args.seed)
    rows = []
    for sec in args.durations:
        T = frames_for_seconds(sec)
        mels = rng.standard_normal((1, T, N_MELS)).astype(np.float32)
        mean, p50, p95 = bench_one(sess, in_name, mels, args.n_runs, args.warmup)
        audio_ms = T * HOP_MS
        rtf = (mean / 1000.0) / (audio_ms / 1000.0)
        rows.append({"duration_s": round(sec, 2), "frames": T,
                     "mean_ms": round(mean, 3), "p50_ms": round(p50, 3),
                     "p95_ms": round(p95, 3), "rtf": round(rtf, 4)})
    return rows


def _load_labels(wav_dir: Path) -> dict:
    labels = {}
    for cand in ("labels.csv", "labels.tsv"):
        p = wav_dir / cand
        if p.exists():
            delim = "," if cand.endswith(".csv") else "\t"
            with p.open("r", encoding="utf-8", newline="") as fh:
                for parts in csv.reader(fh, delimiter=delim):
                    if not parts or not parts[0] or parts[0].lower().startswith("path"):
                        continue
                    if len(parts) >= 2:
                        labels[Path(parts[0]).name] = {
                            "intent": parts[1].strip(),
                            "transcript": parts[2].strip() if len(parts) > 2 else "",
                        }
            break
    return labels


def _summarize_quality(examples: list[dict]) -> dict | None:
    if not examples:
        return None

    confusion = {intent: {prediction: 0 for prediction in INTENTS}
                 for intent in INTENTS}
    correct = 0
    slot_tp = slot_fp = slot_fn = 0
    slot_exact = slot_exact_total = 0
    oov_correct = oov_total = 0
    non_oov_rejected = non_oov_total = 0

    for example in examples:
        ref_intent = example["ref_intent"]
        pred_intent = example["pred_intent"]
        ref_slots = example["ref_slots"]
        pred_slots = example["pred_slots"]
        correct += int(ref_intent == pred_intent)
        confusion[ref_intent][pred_intent] += 1

        if ref_intent == OOV:
            oov_total += 1
            oov_correct += int(pred_intent == OOV)
        else:
            non_oov_total += 1
            non_oov_rejected += int(pred_intent == OOV)

        if ref_slots is None:
            continue
        slot_exact_total += 1
        slot_exact += int(pred_slots == ref_slots)
        for key in set(ref_slots) | set(pred_slots):
            if ref_slots.get(key) == pred_slots.get(key):
                continue
            slot_fn += int(key in ref_slots)
            slot_fp += int(key in pred_slots)
        slot_tp += sum(ref_slots[key] == pred_slots.get(key) for key in ref_slots)

    slot_precision = slot_tp / (slot_tp + slot_fp) if slot_tp + slot_fp else 1.0
    slot_recall = slot_tp / (slot_tp + slot_fn) if slot_tp + slot_fn else 1.0
    slot_f1 = (2 * slot_precision * slot_recall / (slot_precision + slot_recall)
               if slot_precision + slot_recall else 0.0)
    per_intent = {}
    supported_f1 = []
    for intent in INTENTS:
        support = sum(confusion[intent].values())
        predicted = sum(confusion[ref][intent] for ref in INTENTS)
        true_positive = confusion[intent][intent]
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        class_f1 = (2 * precision * recall / (precision + recall)
                    if precision + recall else 0.0)
        if support:
            supported_f1.append(class_f1)
        per_intent[intent] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(class_f1, 4),
        }

    n_labeled = len(examples)
    return {"n_labeled": n_labeled,
            "intent_accuracy": round(correct / n_labeled, 4),
            "intent_macro_f1": round(float(np.mean(supported_f1)), 4),
            "per_intent": per_intent,
            "confusion_matrix": confusion,
            "slot_f1": round(slot_f1, 4),
            "slot_exact_match": (round(slot_exact / slot_exact_total, 4)
                                 if slot_exact_total else None),
            "oov_recall": (round(oov_correct / oov_total, 4)
                           if oov_total else None),
            "non_oov_false_reject_rate": (round(non_oov_rejected / non_oov_total, 4)
                                           if non_oov_total else None)}


def run_wavs(sess, in_name, args):
    wavs = sorted(Path(args.wav_dir).rglob("*.wav"))
    if not wavs:
        raise SystemExit(f"no .wav found under {args.wav_dir}")
    labels = _load_labels(Path(args.wav_dir))
    missing_labels = [path.name for path in wavs if path.name not in labels]
    if args.require_labels and missing_labels:
        raise SystemExit(f"missing labels for {len(missing_labels)} WAVs: {missing_labels[:10]}")
    missing_transcripts = [path.name for path in wavs
                           if not labels.get(path.name, {}).get("transcript")]
    if args.require_labels and missing_transcripts:
        raise SystemExit(f"missing transcripts for {len(missing_transcripts)} WAVs: "
                         f"{missing_transcripts[:10]}")

    rows = []
    quality_examples = []
    for wav_path in wavs:
        wav = load_wav_mono(str(wav_path))
        started = time.perf_counter()
        features = kaldi_fbank(wav)
        feature_ms = (time.perf_counter() - started) * 1000.0
        mels = features[np.newaxis, :, :].astype(np.float32)
        frame_count = features.shape[0]
        mean, p50, p95 = bench_one(sess, in_name, mels, args.n_runs, args.warmup)
        audio_ms = frame_count * HOP_MS
        rtf = mean / audio_ms
        feature_plus_model_ms = feature_ms + mean

        inputs = _make_inputs(sess, in_name, mels)
        outputs = sess.run(None, inputs)
        intent_logits = outputs[0]
        if len(outputs) == 2:
            transcript = " ".join(ctc_greedy_decode(outputs[1][0]))
        else:
            transcript = ""
        pred_intent = INTENTS[int(np.argmax(intent_logits[0]))] if int(np.argmax(intent_logits[0])) < len(INTENTS) else f"intent_{int(np.argmax(intent_logits[0]))}"
        pred_conf = float(softmax(intent_logits[0]).max())
        pred_slots = parse_slots(pred_intent, transcript) if transcript else {}
        row = {"file": wav_path.name, "duration_s": round(len(wav) / SR, 2),
               "frames": frame_count, "mean_ms": round(mean, 3),
               "p50_ms": round(p50, 3), "p95_ms": round(p95, 3),
               "rtf": round(rtf, 4), "mel_ms": round(feature_ms, 2),
               "feature_plus_model_ms": round(feature_plus_model_ms, 3),
               "feature_plus_model_rtf": round(feature_plus_model_ms / audio_ms, 4),
               "pred_intent": pred_intent, "conf": round(pred_conf, 3),
               "transcript": transcript[:40]}

        label = labels.get(wav_path.name)
        if label and label.get("intent"):
            ref_intent = label["intent"]
            if ref_intent not in INTENTS:
                raise SystemExit(f"unknown intent label {ref_intent!r} for {wav_path.name}")
            ref_transcript = (label.get("transcript") or "").lower()
            ref_slots = parse_slots(ref_intent, ref_transcript) if ref_transcript else None
            row["ref_intent"] = ref_intent
            row["intent_ok"] = ref_intent == pred_intent
            quality_examples.append({"ref_intent": ref_intent,
                                     "pred_intent": pred_intent,
                                     "ref_slots": ref_slots,
                                     "pred_slots": pred_slots})
        rows.append(row)

    return rows, _summarize_quality(quality_examples)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    _default_model = resolve_latest_onnx(DEFAULT_ONNX_DIR, "int8")
    _default_model = str(_default_model) if _default_model else str(ROOT / "vcm_model_int8.onnx")
    ap.add_argument("--model", default=_default_model)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--n-runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--durations", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--wav-dir", default=None)
    ap.add_argument("--require-labels", action="store_true",
                    help="fail if any WAV under --wav-dir lacks a label row")
    ap.add_argument("--soak", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--output", default=None,
                    help="optional path to write the full JSON report")
    args = ap.parse_args()
    args.durations = [float(x) for x in str(args.durations).split(",") if x.strip()]

    model = Path(args.model)
    if not model.exists():
        print(f"error: model not found: {model}", file=sys.stderr)
        return 2
    size_mb = model.stat().st_size / 1e6

    so = SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    env_before = env_snapshot()

    if args.wav_dir:
        rows, quality = run_wavs(sess, in_name, args)
    else:
        rows = run_synthetic(sess, in_name, args)
        quality = None

    soak = None
    if args.soak > 0:
        T = frames_for_seconds(1.0)
        dummy = np.random.default_rng(1).standard_normal(
            (1, T, N_MELS)).astype(np.float32)
        soak_inputs = _make_inputs(sess, in_name, dummy)
        t_end = time.time() + args.soak
        while time.time() < t_end:
            sess.run(None, soak_inputs)
        env_after_soak = env_snapshot()
        mean, p50, p95 = bench_one(sess, in_name, dummy, args.n_runs, args.warmup)
        audio_ms = T * HOP_MS
        soak = {"soak_s": args.soak,
                "clock_before_ghz": env_before.get("arm_clock_ghz"),
                "clock_after_ghz": env_after_soak.get("arm_clock_ghz"),
                "temp_after_c": env_after_soak.get("temp_c"),
                "throttled": env_after_soak.get("throttled"),
                "post_soak_mean_ms": round(mean, 3),
                "post_soak_p95_ms": round(p95, 3),
                "post_soak_rtf": round((mean / 1000.0) / (audio_ms / 1000.0), 4)}

    env_after = env_snapshot()
    out = {"model": str(model), "model_sha256": file_sha256(model),
           "model_size_mb": round(size_mb, 3),
           "threads": args.threads,
           "env_before": env_before, "env_after": env_after,
           "peak_rss_mb": round(peak_rss_mb(), 1),
           "rows": rows, "quality": quality, "soak": soak}

    typ = [r for r in rows if 0.9 <= r["duration_s"] <= 2.05]
    worst_rtf = max((r["rtf"] for r in rows), default=None)
    p95_typ = max((r["p95_ms"] for r in typ), default=None)
    out["pass"] = {"rtf_le_0.3": (worst_rtf is not None and worst_rtf <= 0.3),
                   "latency_p95_lt_100ms": (p95_typ is not None and p95_typ < 100.0),
                   "intent_acc_ge_0.9": (quality["intent_accuracy"] >= 0.9
                                         if quality and quality.get("intent_accuracy") is not None
                                         else None)}

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"model      : {model.name}  ({size_mb:.2f} MB; size is informational)")
        print(f"threads    : {args.threads}   peak RSS: {out['peak_rss_mb']} MB")
        eb, ea = env_before, env_after
        if "arm_clock_ghz" in eb:
            print(f"clock      : {eb['arm_clock_ghz']} GHz -> "
                  f"{ea.get('arm_clock_ghz')} GHz   temp {ea.get('temp_c','?')}")
        print("-" * 78)
        print(f"{'dur(s)':>7} {'frames':>7} {'mean(ms)':>10} {'p50(ms)':>9} "
              f"{'p95(ms)':>9} {'RTF':>7}")
        for r in rows:
            p50 = r.get('p50_ms', r['mean_ms'])
            print(f"{r['duration_s']:>7} {r['frames']:>7} {r['mean_ms']:>10} "
                  f"{p50:>9} {r['p95_ms']:>9} {r['rtf']:>7}")
        if quality:
            print("-" * 78)
            print(f"QUALITY    : intent_acc={quality['intent_accuracy']}  "
                                    f"macro_F1={quality['intent_macro_f1']}  "
                                    f"slot_F1={quality['slot_f1']}  "
                                    f"slot_exact={quality['slot_exact_match']}  "
                                    f"oov_recall={quality['oov_recall']}  "
                  f"(n={quality['n_labeled']})")
        if soak:
            print("-" * 78)
            print(f"SOAK {soak['soak_s']}s : clock {soak['clock_before_ghz']}->"
                  f"{soak['clock_after_ghz']} GHz  temp {soak['temp_after_c']}  "
                  f"throttled={soak['throttled']}")
            print(f"           post-soak 1s cmd: mean {soak['post_soak_mean_ms']} ms, "
                  f"p95 {soak['post_soak_p95_ms']} ms, RTF {soak['post_soak_rtf']}")
        p = out["pass"]
        print("-" * 78)
        print(f"PASS rtf<=0.3={p['rtf_le_0.3']}  "
              f"p95<100ms={p['latency_p95_lt_100ms']}  "
              f"acc>=0.9={p['intent_acc_ge_0.9']}")
        if args.output:
            print(f"report     : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
