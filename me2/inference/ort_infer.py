"""Torch-free on-device inference for the Pi5-VCM voice command model.

Drop-in replacement for ``inference.infer`` that runs on ONNX Runtime with
ONLY ``numpy`` + ``onnxruntime`` installed -- no torch, no torchaudio. This
is the path the Raspberry Pi 5 harness uses: the Pi's venv ships exactly
those two packages and nothing else.

Why a separate module
---------------------
``inference.infer`` loads ``best.pt`` through PyTorch and computes features
with ``torchaudio.compliance.kaldi.fbank``. Neither is available on the Pi.
Here the model is the int8-quantized ONNX export (``vcm_model_int8.onnx``)
and features come from :mod:`inference.features.kaldi_fbank`, a faithful
numpy port of the *same* ``kaldi.fbank`` the model was trained on (validated
to float32 precision). The result: identical intents/slots/transcripts to
the torch path, at a fraction of the memory footprint.

Public API (mirrors ``inference.infer``)
----------------------------------------
resolve_checkpoint(raw)                 -> Path
load_session(checkpoint, threads=1)     -> (session, in_name, intents, ctc_vocab)
run_utterance(session, wav, max_frames, intents, ctc_vocab) -> dict
self_test_wavs()                        -> list[(name, wav_np)]
main(argv)                              -> int   (CLI, like infer.py)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from onnxruntime import InferenceSession, SessionOptions
from onnxruntime.capi.onnxruntime_pybind11_state import GraphOptimizationLevel

# The canonical project modules (config, model/) live at the project root, one
# level above this folder. Put the root on sys.path so they resolve regardless
# of the current working directory. There are no vendored copies here.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from model.model import parse_slots  # noqa: E402  (pure-python, no torch)
try:
    from .features import kaldi_fbank, load_wav_mono, synthesize_utterance  # noqa: E402
except ImportError:  # direct script run (python inference/ort_infer.py)
    from features import kaldi_fbank, load_wav_mono, synthesize_utterance  # noqa: E402


# ---------------------------------------------------------------------------
# checkpoint / session
# ---------------------------------------------------------------------------
def resolve_checkpoint(raw: str) -> Path:
    """Resolve the checkpoint path, trying this folder then the project root."""
    p = Path(raw)
    if p.is_absolute():
        return p
    for base in (ROOT, ROOT.parent):
        cand = base / p
        if cand.exists():
            return cand
    return ROOT / p


def load_session(checkpoint: str | Path, threads: int = 1):
    """Load the ONNX model -> ``(session, input_name, intents, ctc_vocab)``.

    ``threads`` maps to ``intra_op_num_threads`` (ORT parallelism for the
    intra-op graph). ``inter_op`` is kept at 1 -- the model is a single
    sequential chain, so inter-op parallelism buys nothing.
    """
    ckpt = Path(checkpoint)
    opts = SessionOptions()
    opts.intra_op_num_threads = max(1, int(threads))
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = InferenceSession(str(ckpt), opts,
                            providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    return sess, in_name, list(config.INTENTS), list(config.CTC_VOCAB)


# ---------------------------------------------------------------------------
# decode (numpy)
# ---------------------------------------------------------------------------
def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def ctc_greedy_decode(ctc_logits: np.ndarray,
                      vocab: list[str] | None = None) -> list[str]:
    """Greedy CTC decode of one utterance: (T, V) -> list of tokens.

    argmax per frame -> collapse repeats -> drop blanks. Identical to
    ``model.model.ctc_greedy_decode`` but numpy.
    """
    if vocab is None:
        vocab = config.CTC_VOCAB
    ids = np.argmax(ctc_logits, axis=-1)
    tokens: list[str] = []
    prev = -1
    for i in ids.tolist():
        if i != prev and i != config.CTC_BLANK:
            tokens.append(vocab[i])
        prev = i
    return tokens


def _trim_frames(mel: np.ndarray, max_frames: int) -> np.ndarray:
    """Trim (never zero-pad) to at most ``max_frames`` rows.

    True-length inference: a 1.2 s utterance runs ~120 frames, not 400. This
    is the main on-device latency lever for the "instant" requirement.
    """
    if mel.shape[0] > max_frames:
        return mel[:max_frames]
    return mel


# ---------------------------------------------------------------------------
# main inference
# ---------------------------------------------------------------------------
def run_utterance(session, wav: np.ndarray, max_frames: int,
                  intents: list, ctc_vocab: list) -> dict:
    """wav: (samples,) float32 @16 kHz -> result dict.

    Returns the SAME dict shape as ``inference.infer.run_utterance`` so the
    harness/dispatcher/reply code is unchanged:
        {intent, intent_confidence, transcript, slots, frames, latency_ms}
    """
    wav = np.asarray(wav, dtype=np.float32).ravel()
    mel = kaldi_fbank(wav)                     # (T, 80)
    mel = _trim_frames(mel, max_frames)
    if mel.shape[0] == 0:
        mel = np.zeros((1, config.N_MELS), dtype=np.float32)
    x = mel[np.newaxis, :, :].astype(np.float32)   # (1, T, 80)

    t0 = time.perf_counter()
    intent_logits, ctc_logits = session.run(None, {"mels": x})
    latency_ms = (time.perf_counter() - t0) * 1000.0

    probs = _softmax(intent_logits[0])
    top_id = int(np.argmax(probs))
    intent = intents[top_id]
    tokens = ctc_greedy_decode(ctc_logits[0], ctc_vocab)
    transcript = " ".join(tokens)
    slots = parse_slots(intent, transcript)
    return {
        "intent": intent,
        "intent_confidence": round(float(probs[top_id]), 4),
        "transcript": transcript,
        "slots": slots,
        "frames": int(mel.shape[0]),
        "latency_ms": round(latency_ms, 2),
    }


def self_test_wavs():
    """A few toy utterances so the pipeline can be smoke-tested without real
    recordings. These are formant tones, NOT real speech (numpy only)."""
    cases = [
        (["turn", "on", "the", "lights"], "spk_a"),
        (["set", "timer", "for", "5", "minutes"], "spk_b"),
        (["what", "is", "the", "weather"], "spk_a"),
        (["call", "mom"], "spk_c"),
    ]
    items = []
    for i, (words, spk) in enumerate(cases):
        wav_np = synthesize_utterance(words, spk, seed=i)
        items.append((f"selftest:{' '.join(words)}", wav_np))
    return items


def warmup(session, n: int = 3) -> None:
    """Absorb first-call init / allocator cost so reported latency is
    steady-state (the number that matters for "instant")."""
    dummy = np.zeros((1, 100, config.N_MELS), dtype=np.float32)
    for _ in range(max(0, int(n))):
        session.run(None, {"mels": dummy})


# ---------------------------------------------------------------------------
# CLI (parity with infer.py)
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Pi5-VCM on-device inference (input: vcm_model_int8.onnx).")
    ap.add_argument("--checkpoint", default="../vcm_model_int8.onnx",
                    help="path to the ONNX model (default: ../vcm_model_int8.onnx)")
    ap.add_argument("--input", help="path to a 16 kHz mono wav file")
    ap.add_argument("--self-test", action="store_true",
                    help="run built-in synthetic utterances (no wav needed)")
    ap.add_argument("--threads", type=int, default=1,
                    help="ORT intra_op_num_threads (default 1; use 2 on Pi5)")
    ap.add_argument("--max-frames", type=int, default=400,
                    help="max 10 ms frames (400 = 4 s); trimmed, not padded")
    ap.add_argument("--warmup", type=int, default=3,
                    help="warmup forward passes before timing (default 3)")
    ap.add_argument("--json", action="store_true", help="emit JSON on stdout")
    args = ap.parse_args(argv)

    ckpt = resolve_checkpoint(args.checkpoint)
    if not ckpt.exists():
        print(f"error: checkpoint not found: {ckpt}", file=sys.stderr)
        return 2

    sess, in_name, intents, ctc_vocab = load_session(ckpt, args.threads)
    warmup(sess, args.warmup)

    if args.input:
        wav = load_wav_mono(args.input)
        items = [(Path(args.input).name, wav)]
    elif args.self_test:
        items = self_test_wavs()
    else:
        print("error: provide --input <wav> or --self-test", file=sys.stderr)
        return 2

    results = []
    for name, wav in items:
        r = run_utterance(sess, wav, args.max_frames, intents, ctc_vocab)
        r["input"] = name
        results.append(r)

    if args.json:
        print(json.dumps({"backend": "onnxruntime",
                          "checkpoint": str(ckpt),
                          "threads": args.threads,
                          "results": results}, indent=2))
    else:
        print(f"backend=onnxruntime  checkpoint={ckpt}  threads={args.threads}")
        for r in results:
            print("-" * 60)
            print(f"  input     : {r['input']}")
            print(f"  intent    : {r['intent']}  (conf {r['intent_confidence']:.3f})")
            print(f"  transcript: {r['transcript']!r}")
            print(f"  slots     : {r['slots']}")
            print(f"  frames    : {r['frames']}  ({r['frames'] * 10} ms)")
            print(f"  latency   : {r['latency_ms']} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
