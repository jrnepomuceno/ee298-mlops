#!/usr/bin/env python3
"""Int8 verification for the deployed VCM ONNX (v19-20260929-2).

Three checks in one pass, torch-free (onnxruntime + numpy only):

1. INT8 ACCURACY (fidelity)  -- the real meaning of "int8 accuracy" for a
   shipped model: run the SAME mel inputs through the FP32 graph (reference)
   and the dynamic-int8 graph (deployment) and measure the degradation.
   Reports argmax agreement on intent + CTC, cosine similarity, and L2.
   Uses BOTH deterministic random mels AND the real reply-audio mels so the
   check covers in-distribution-ish signal, not just noise.

2. RTF                      -- compute_time / audio_time, per duration, for a
   1-2 s command (the pass bar is RTF <= 0.3).

3. P95 LATENCY              -- per-utterance session.run() wall time,
   p50/p95/max, per duration (pass bar p95 < 100 ms for 1-2 s).

Timing scope is session.run() only (isolates the model). Mel extraction is
timed separately so the full path is visible.

Usage:
  python3 scripts/int8_check.py [--tag v19-20260929-2] [--runs 200] [--threads 1]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
from onnxruntime import InferenceSession, SessionOptions

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from model.onnx_deploy import DEFAULT_ONNX_DIR, resolve_latest_onnx  # noqa: E402

SR = config.SAMPLE_RATE
N_MELS = config.N_MELS
HOP_MS = config.FRAME_SHIFT_MS
FRAME_MS = getattr(config, "FRAME_LENGTH_MS", 25)


# --------------------------------------------------------------------------- io
def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() in (1, 2)
        sw = w.getsampwidth()
        assert sw in (1, 2, 4)
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
        new_n = max(1, int(round(dur * SR)))
        xo = np.linspace(0, 1, len(arr), endpoint=False)
        xn = np.linspace(0, 1, new_n, endpoint=False)
        arr = np.interp(xn, xo, arr)
    return arr.astype(np.float32)


def mel_from_wav(wav: np.ndarray) -> np.ndarray:
    """Log-mel (T, 80) via numpy STFT + mel filterbank (shape-contract match)."""
    n_fft = int(SR * FRAME_MS / 1000.0)          # 400
    hop = int(SR * HOP_MS / 1000.0)              # 160
    if len(wav) < n_fft:
        wav = np.pad(wav, (0, n_fft - len(wav)))
    n_frames = 1 + (len(wav) - n_fft) // hop
    idx = (np.arange(n_frames)[:, None] * hop + np.arange(n_fft)[None, :])
    frames = wav[idx]
    window = np.hanning(n_fft).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    # mel filterbank
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    fmin, fmax = 0.0, SR / 2.0
    mel_pts = np.linspace(hz2mel(fmin), hz2mel(fmax), N_MELS + 2)
    hz_pts = mel2hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / SR).astype(int)
    fb = np.zeros((N_MELS, n_fft // 2 + 1))
    for i in range(N_MELS):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        for j in range(l, c):
            if c > l:
                fb[i, j] = (j - l) / (c - l)
        for j in range(c, r):
            if r > c:
                fb[i, j] = (r - j) / (r - c)
    mel = spec @ fb.T
    mel = np.log(mel + 1e-10)
    return mel.astype(np.float32)


def random_mels(T: int, batch: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # log-mel-like magnitude range
    return (rng.standard_normal((batch, T, N_MELS)) * 2.0 - 4.0).astype(np.float32)


def ctc_decode(logits: np.ndarray) -> list[int]:
    ids = logits.argmax(axis=-1)
    out, prev = [], -1
    blank = int(config.CTC_BLANK)
    for i in ids:
        if i != prev and i != blank:
            out.append(int(i))
        prev = i
    return out


# --------------------------------------------------------------------------- run
def make_session(path: Path, threads: int) -> InferenceSession:
    from onnxruntime.capi.onnxruntime_pybind11_state import (
        GraphOptimizationLevel, ExecutionMode,
    )
    opts = SessionOptions()
    opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ExecutionMode.ORT_SEQUENTIAL
    return InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="version tag dir (default: resolve latest)")
    ap.add_argument("--runs", type=int, default=200, help="timed iterations per duration")
    ap.add_argument("--threads", type=int, default=1, help="ORT intra-op threads (1 = Pi-like)")
    ap.add_argument("--durations", default="0.5,1.0,1.5,2.0,3.0", help="seconds, comma-sep")
    args = ap.parse_args()

    if args.tag:
        base = DEFAULT_ONNX_DIR / args.tag
        fp32 = base / "vcm_model.onnx"
        int8 = base / "vcm_model_int8.onnx"
    else:
        int8 = resolve_latest_onnx(kind="int8")
        fp32 = int8.with_name("vcm_model.onnx")
    assert fp32.exists(), f"missing fp32: {fp32}"
    assert int8.exists(), f"missing int8: {int8}"
    tag = int8.parent.name

    print(f"=== int8 check :: {tag} ===")
    print(f"fp32 : {fp32}")
    print(f"int8 : {int8}")
    print(f"threads={args.threads}  runs={args.runs}  provider=CPUExecutionProvider")

    sess_fp32 = make_session(fp32, args.threads)
    sess_int8 = make_session(int8, args.threads)

    # ---- build eval set: random mels (several T) + real reply-audio mels ----
    eval_inputs: list[np.ndarray] = []
    for T in (25, 50, 100, 160, 240):
        eval_inputs.append(random_mels(T, batch=1, seed=T))
    reply_dir = ROOT / "assets" / "replies"
    if reply_dir.exists():
        for w in sorted(reply_dir.glob("*.wav")):
            wav = load_wav(w)
            m = mel_from_wav(wav)
            if m.shape[0] >= 5:
                eval_inputs.append(m[None, :, :])
    print(f"\neval inputs: {len(eval_inputs)} (random-mel + {max(0,len(eval_inputs)-5)} real reply wavs)")

    # =========================================================================
    # 1) INT8 ACCURACY (fidelity vs fp32 reference)
    # =========================================================================
    intent_agree, ctc_agree = 0, 0
    cos_intent, cos_ctc = [], []
    l2_intent, l2_ctc = [], []
    n = 0
    for x in eval_inputs:
        a_i, a_c = sess_fp32.run(None, {"mels": x})
        b_i, b_c = sess_int8.run(None, {"mels": x})
        # intent argmax agreement
        ai, bi = a_i.argmax(-1), b_i.argmax(-1)
        intent_agree += int((ai == bi).all())
        # ctc argmax-token sequence agreement
        ctc_agree += int(ctc_decode(a_c[0]) == ctc_decode(b_c[0]))
        # cosine + l2
        af, bf = a_i.reshape(-1), b_i.reshape(-1)
        cos_intent.append(float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12)))
        l2_intent.append(float(np.linalg.norm(af - bf)))
        ac, bc = a_c.reshape(-1), b_c.reshape(-1)
        cos_ctc.append(float(np.dot(ac, bc) / (np.linalg.norm(ac) * np.linalg.norm(bc) + 1e-12)))
        l2_ctc.append(float(np.linalg.norm(ac - bc)))
        n += 1

    print("\n--- 1) INT8 ACCURACY (fidelity vs FP32 reference) ---")
    print(f"inputs compared            : {n}")
    print(f"intent argmax agreement    : {intent_agree}/{n} = {100.0*intent_agree/n:.2f}%")
    print(f"ctc token-seq agreement    : {ctc_agree}/{n} = {100.0*ctc_agree/n:.2f}%")
    print(f"intent logits cos (mean)   : {np.mean(cos_intent):.6f}  (min {np.min(cos_intent):.6f})")
    print(f"ctc logits cos (mean)      : {np.mean(cos_ctc):.6f}  (min {np.min(cos_ctc):.6f})")
    print(f"intent logits L2 (mean)    : {np.mean(l2_intent):.5f}  (max {np.max(l2_intent):.5f})")
    print(f"ctc logits L2 (mean)       : {np.mean(l2_ctc):.5f}  (max {np.max(l2_ctc):.5f})")

    # =========================================================================
    # 2) + 3) RTF and P95 LATENCY  (int8, the deployment graph)
    # =========================================================================
    durations = [float(d) for d in args.durations.split(",")]
    print("\n--- 2/3) RTF + LATENCY (int8, session.run only) ---")
    hdr = f"{'dur(s)':>7} {'T':>5} {'run_ms_mean':>12} {'p50':>8} {'p95':>8} {'max':>8} {'rtf':>8} {'pass_rtf':>9} {'pass_p95':>9}"
    print(hdr)
    print("-" * len(hdr))
    perf = {}
    for d in durations:
        T = int(round(d * 1000.0 / HOP_MS))
        x = random_mels(T, batch=1, seed=1000 + int(d * 10))
        # warmup
        for _ in range(10):
            sess_int8.run(None, {"mels": x})
        lat = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            sess_int8.run(None, {"mels": x})
            lat.append((time.perf_counter() - t0) * 1000.0)
        lat = np.array(lat)
        mean_ms = float(lat.mean())
        p50 = float(np.percentile(lat, 50))
        p95 = float(np.percentile(lat, 95))
        mx = float(lat.max())
        rtf = mean_ms / (d * 1000.0)
        perf[f"{d}s"] = {"T": T, "mean_ms": round(mean_ms, 3), "p50_ms": round(p50, 3),
                         "p95_ms": round(p95, 3), "max_ms": round(mx, 3), "rtf": round(rtf, 4)}
        pr = "OK" if rtf <= 0.3 else "FAIL"
        pp = "OK" if p95 < 100.0 else "FAIL"
        print(f"{d:>7.1f} {T:>5} {mean_ms:>12.3f} {p50:>8.2f} {p95:>8.2f} {mx:>8.2f} {rtf:>8.4f} {pr:>9} {pp:>9}")

    # mel-extraction cost (full-path visibility), 1.5 s clip
    wav = random_mels(1, 1, 0).reshape(-1)  # placeholder length
    clip = np.zeros(int(1.5 * SR), dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(50):
        mel_from_wav(clip)
    mel_ms = (time.perf_counter() - t0) / 50 * 1000.0
    print(f"\nmel extraction (1.5s clip, numpy, 50 iters): {mel_ms:.2f} ms  "
          f"(full path ~= run + {mel_ms:.0f} ms)")

    # ---- summary verdict ----
    ok_intent = intent_agree == n
    ok_ctc = ctc_agree == n
    worst_rtf = max(v["rtf"] for v in perf.values())
    worst_p95 = max(v["p95_ms"] for v in perf.values())
    print("\n=== VERDICT ===")
    print(f"int8 fidelity : intent {'PASS' if ok_intent else 'DEGRADED'} "
          f"({100.0*intent_agree/n:.2f}%), ctc {'PASS' if ok_ctc else 'DEGRADED'} "
          f"({100.0*ctc_agree/n:.2f}%)")
    print(f"RTF           : worst {worst_rtf:.4f} -> {'PASS (<=0.3)' if worst_rtf<=0.3 else 'FAIL'}")
    print(f"P95 latency   : worst {worst_p95:.2f} ms -> {'PASS (<100ms)' if worst_p95<100 else 'FAIL'}")

    out = {
        "tag": tag, "threads": args.threads, "runs": args.runs,
        "provider": "CPUExecutionProvider", "platform": sys.platform,
        "fidelity": {
            "n_inputs": n,
            "intent_argmax_agreement": round(intent_agree / n, 4),
            "ctc_token_seq_agreement": round(ctc_agree / n, 4),
            "intent_cos_mean": round(float(np.mean(cos_intent)), 6),
            "ctc_cos_mean": round(float(np.mean(cos_ctc)), 6),
            "intent_l2_mean": round(float(np.mean(l2_intent)), 5),
            "ctc_l2_mean": round(float(np.mean(l2_ctc)), 5),
        },
        "perf": perf,
        "mel_extract_ms_1p5s": round(mel_ms, 2),
    }
    out_path = ROOT / "models" / "onnx" / tag / "int8_check.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
