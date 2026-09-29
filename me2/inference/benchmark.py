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

Pass bars (docs/plan.md): RTF <= 0.3, latency p95 < 100 ms for a 1-2 s command,
int8 <= 6 MB (checked at load).
"""
from __future__ import annotations

import argparse
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


def parse_slots(intent: str, transcript: str) -> dict:
    """Mirror of model.model.parse_slots (rule-based, no LLM)."""
    import re
    slots: dict = {}
    t = transcript.lower()
    t = re.sub(r"(?<=\d) (?=\d)", "", t)   # re-join digit tokens ("1 8" -> "18")
    m = re.search(r"(\d+)\s*percent", t)
    if m:
        slots["percent"] = int(m.group(1))
    m = re.search(r"(\d+)\s*degrees?", t)
    if m:
        slots["temperature"] = int(m.group(1))
    m = re.search(r"(\d+)\s*(minutes?|seconds?)", t)
    if m:
        slots["duration"] = int(m.group(1))
        slots["duration_unit"] = m.group(2).rstrip("s")
    m = re.search(r"(\d+)\s*(am|pm)", t)
    if m:
        slots["time"] = f"{m.group(1)}:00 {m.group(2)}"
    return slots


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


def bench_one(sess, in_name, mels, n_runs, warmup):
    """Time session.run() only. Returns (mean_ms, p50_ms, p95_ms)."""
    for _ in range(max(0, warmup)):
        sess.run(None, {in_name: mels})
    lats = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {in_name: mels})
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


def _hz_to_mel(hz: float) -> float:
    return 1127.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 1127.0) - 1.0)


def _load_wav_np(path: str) -> np.ndarray:
    import wave
    with wave.open(path, "rb") as w:
        assert w.getnchannels() in (1, 2), "need mono/stereo wav"
        sw = w.getsampwidth()
        assert sw in (1, 2, 4), f"unsupported sampwidth {sw}"
        n = w.getnframes()
        raw = w.readframes(n)
        sr = w.getframerate()
        fmt = {1: "int8", 2: "int16", 4: "int32"}[sw]
        arr = np.frombuffer(raw, dtype=fmt).astype(np.float64)
        if w.getnchannels() == 2:
            arr = arr.reshape(-1, 2).mean(axis=1)
        scale = {1: 127.0, 2: 32767.0, 4: 2147483647.0}[sw]
        arr = arr / scale
    if sr != SR:
        dur = len(arr) / sr
        new_n = int(round(dur * SR))
        x_old = np.linspace(0, 1, len(arr), endpoint=False)
        x_new = np.linspace(0, 1, new_n, endpoint=False)
        arr = np.interp(x_new, x_old, arr).astype(np.float32)
    return arr.astype(np.float32)


def _mel_from_wav(wav: np.ndarray) -> np.ndarray:
    """Log-mel (T, 80) with a numpy fbank (no torch). Shape-contract match."""
    n_fft = int(SR * config.FRAME_LENGTH_MS / 1000.0)   # 400
    hop = int(SR * HOP_MS / 1000.0)                     # 160
    if len(wav) < n_fft:
        wav = np.pad(wav, (0, n_fft - len(wav)))
    n_frames = 1 + (len(wav) - n_fft) // hop
    idx = (np.arange(n_frames)[:, None] * hop + np.arange(n_fft)[None, :])
    frames = wav[idx]
    window = np.hamming(n_fft).astype(np.float32)
    frames = frames * window
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    spec = spec + 1e-10
    n_bins = n_fft // 2 + 1
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SR)
    mel_lo, mel_hi = _hz_to_mel(0.0), _hz_to_mel(SR / 2.0)
    mels_hz = _mel_to_hz(np.linspace(mel_lo, mel_hi, N_MELS + 2))
    filters = np.zeros((N_MELS, n_bins))
    for i in range(N_MELS):
        lo, c, hi = mels_hz[i], mels_hz[i + 1], mels_hz[i + 2]
        for j in range(n_bins):
            f = freqs[j]
            if lo < f < c:
                filters[i, j] = (f - lo) / (c - lo)
            elif c < f < hi:
                filters[i, j] = (hi - f) / (hi - c)
    feat = spec @ filters.T
    feat = np.log(feat + 1e-10).astype(np.float32)
    return feat


def _load_labels(wav_dir: Path) -> dict:
    labels = {}
    for cand in ("labels.csv", "labels.tsv"):
        p = wav_dir / cand
        if p.exists():
            delim = "," if cand.endswith(".csv") else "\t"
            with open(p) as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln or ln.lower().startswith("path"):
                        continue
                    parts = ln.split(delim)
                    if len(parts) >= 2:
                        labels[Path(parts[0]).name] = {
                            "intent": parts[1].strip(),
                            "transcript": parts[2].strip() if len(parts) > 2 else "",
                        }
            break
    return labels


def run_wavs(sess, in_name, args):
    wavs = sorted(Path(args.wav_dir).rglob("*.wav"))
    if not wavs:
        raise SystemExit(f"no .wav found under {args.wav_dir}")
    labels = _load_labels(Path(args.wav_dir))
    rows = []
    correct = 0
    n_labeled = 0
    slot_tp = slot_fp = slot_fn = 0
    oov_correct = 0
    oov_total = 0
    for wp in wavs:
        wav = _load_wav_np(str(wp))
        t0 = time.perf_counter()
        feat = _mel_from_wav(wav)
        mel_ms = (time.perf_counter() - t0) * 1000.0
        mels = feat[np.newaxis, :, :].astype(np.float32)
        T = feat.shape[0]
        mean, p50, p95 = bench_one(sess, in_name, mels, args.n_runs, args.warmup)
        audio_ms = T * HOP_MS
        rtf = (mean / 1000.0) / (audio_ms / 1000.0)
        il, cl = sess.run(None, {in_name: mels})
        pred_intent = INTENTS[int(np.argmax(il[0]))]
        pred_conf = float(softmax(il[0]).max())
        tokens = ctc_greedy_decode(cl[0])
        transcript = " ".join(tokens)
        slots = parse_slots(pred_intent, transcript)
        row = {"file": wp.name, "duration_s": round(len(wav) / SR, 2),
               "frames": T, "mean_ms": round(mean, 3), "p95_ms": round(p95, 3),
               "rtf": round(rtf, 4), "mel_ms": round(mel_ms, 2),
               "pred_intent": pred_intent, "conf": round(pred_conf, 3),
               "transcript": transcript[:40]}
        if wp.name in labels:
            lab = labels[wp.name]
            ref_intent = lab.get("intent")
            ref_trans = (lab.get("transcript") or "").lower()
            if ref_intent:
                n_labeled += 1
                ok = (ref_intent == pred_intent)
                correct += int(ok)
                row["ref_intent"] = ref_intent
                row["intent_ok"] = bool(ok)
                if ref_intent == OOV:
                    oov_total += 1
                    oov_correct += int(pred_intent == OOV)
                if ref_trans:
                    ref_slots = parse_slots(ref_intent, ref_trans)
                    for k in set(ref_slots) | set(slots):
                        if ref_slots.get(k) == slots.get(k):
                            continue
                        if k in ref_slots:
                            slot_fn += 1
                        if k in slots:
                            slot_fp += 1
                    for k in ref_slots:
                        if ref_slots[k] == slots.get(k):
                            slot_tp += 1
        rows.append(row)

    quality = None
    if n_labeled:
        prec = slot_tp / (slot_tp + slot_fp) if (slot_tp + slot_fp) else 1.0
        rec = slot_tp / (slot_tp + slot_fn) if (slot_tp + slot_fn) else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        quality = {"n_labeled": n_labeled,
                   "intent_accuracy": round(correct / n_labeled, 4),
                   "slot_f1": round(f1, 4),
                   "oov_rejection": (round(oov_correct / oov_total, 4)
                                     if oov_total else None)}
    return rows, quality


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
    ap.add_argument("--soak", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    args.durations = [float(x) for x in str(args.durations).split(",") if x.strip()]

    model = Path(args.model)
    if not model.exists():
        print(f"error: model not found: {model}", file=sys.stderr)
        return 2
    size_mb = model.stat().st_size / 1e6
    budget_ok = size_mb <= 6.0

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
        t_end = time.time() + args.soak
        while time.time() < t_end:
            sess.run(None, {in_name: dummy})
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
    out = {"model": str(model), "size_mb": round(size_mb, 3),
           "size_budget_ok": budget_ok, "threads": args.threads,
           "env_before": env_before, "env_after": env_after,
           "peak_rss_mb": round(peak_rss_mb(), 1),
           "rows": rows, "quality": quality, "soak": soak}

    typ = [r for r in rows if 0.9 <= r["duration_s"] <= 2.05]
    worst_rtf = max((r["rtf"] for r in rows), default=None)
    p95_typ = max((r["p95_ms"] for r in typ), default=None)
    out["pass"] = {"size": budget_ok,
                   "rtf_le_0.3": (worst_rtf is not None and worst_rtf <= 0.3),
                   "latency_p95_lt_100ms": (p95_typ is not None and p95_typ < 100.0),
                   "intent_acc_ge_0.9": (quality is not None
                                         and quality.get("intent_accuracy") is not None
                                         and quality["intent_accuracy"] >= 0.9)}

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"model      : {model.name}  ({size_mb:.2f} MB, budget<=6MB: "
              f"{'PASS' if budget_ok else 'FAIL'})")
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
                  f"slot_F1={quality['slot_f1']}  "
                  f"oov_reject={quality['oov_rejection']}  "
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
        print(f"PASS size={p['size']}  rtf<=0.3={p['rtf_le_0.3']}  "
              f"p95<100ms={p['latency_p95_lt_100ms']}  "
              f"acc>=0.9={p['intent_acc_ge_0.9']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
