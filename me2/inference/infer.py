#!/usr/bin/env python3
"""On-device inference for the Pi5-VCM voice command model.

Loads a trained checkpoint (``best.pt``) and runs one audio file (or a
built-in self-test) through the model, printing the predicted intent, the
decoded transcript, the rule-based slots, and the per-utterance latency.

Designed for the Raspberry Pi 5: CPU-only, standalone, no cloud, no LLM.
The constrained CTC head + ``parse_slots`` is the LLM replacement.

Usage:
    python infer.py --checkpoint ../checkpoints/best.pt --input cmd.wav
    python infer.py --checkpoint best.pt --self-test
    python infer.py --checkpoint best.pt --input a.wav --json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Make the vendored project modules (config, model, utils) importable
# regardless of the current working directory.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config  # noqa: E402
from model import VCM, ctc_decode_batch, parse_slots  # noqa: E402
from utils import audio_utils  # noqa: E402
from utils.model_utils import get_device  # noqa: E402


def load_model(checkpoint: str, device: torch.device):
    """Load ``best.pt`` -> (model, intents, ctc_vocab, payload).

    The checkpoint payload (see utils/model_utils.save_checkpoint) carries
    ``model_state`` plus a ``config`` block with the intent list and CTC
    vocabulary, so inference always matches how the model was trained.
    """
    payload = torch.load(checkpoint, map_location=device)
    model = VCM()
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    cfg = payload.get("config", {}) or {}
    intents = cfg.get("intents") or config.INTENTS
    ctc_vocab = cfg.get("ctc_vocab") or config.CTC_VOCAB
    return model, intents, ctc_vocab, payload


def trim_frames(mel: torch.Tensor, max_frames: int) -> torch.Tensor:
    """Trim (never zero-pad) to at most ``max_frames`` rows.

    True-length inference: a 1.2 s utterance runs 120 frames, not 400.
    This is the main on-device latency lever for the "instant" requirement.
    """
    if mel.shape[0] > max_frames:
        return mel[:max_frames]
    return mel


@torch.no_grad()
def run_utterance(model, wav: torch.Tensor, device: torch.device,
                  max_frames: int, intents: list, ctc_vocab: list) -> dict:
    """wav: (samples,) float32 @16 kHz -> result dict."""
    mel = audio_utils.mel_spectrogram(wav)          # (T, 80)
    mel = trim_frames(mel, max_frames)
    x = mel.unsqueeze(0).to(device)                # (1, T, 80)

    t0 = time.perf_counter()
    intent_logits, ctc_logits = model(x)
    transcript = ctc_decode_batch(ctc_logits, ctc_vocab)[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    probs = torch.softmax(intent_logits, dim=-1)[0]
    top_id = int(probs.argmax().item())
    intent = intents[top_id]
    slots = parse_slots(intent, transcript)
    return {
        "intent": intent,
        "intent_confidence": round(float(probs[top_id].item()), 4),
        "transcript": transcript,
        "slots": slots,
        "frames": int(mel.shape[0]),
        "latency_ms": round(latency_ms, 2),
    }


def self_test_wavs():
    """A few toy utterances so the pipeline can be smoke-tested without
    real recordings. These are formant tones, NOT real speech."""
    cases = [
        (["turn", "on", "the", "lights"], "spk_a"),
        (["set", "timer", "for", "5", "minutes"], "spk_b"),
        (["what", "is", "the", "weather"], "spk_a"),
        (["call", "mom"], "spk_c"),
    ]
    items = []
    for i, (words, spk) in enumerate(cases):
        wav_np = audio_utils.synthesize_utterance(words, spk, seed=i)
        items.append((f"selftest:{' '.join(words)}",
                      torch.from_numpy(wav_np).float()))
    return items


def resolve_checkpoint(raw: str) -> Path:
    """Resolve the checkpoint path, trying this folder then the project root."""
    p = Path(raw)
    if p.is_absolute():
        return p
    for base in (HERE, HERE.parent):
        cand = base / p
        if cand.exists():
            return cand
    return HERE / p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Pi5-VCM on-device inference (input: best.pt).")
    ap.add_argument("--checkpoint", default="../checkpoints/best.pt",
                    help="path to best.pt (default: ../checkpoints/best.pt)")
    ap.add_argument("--input", help="path to a 16 kHz mono wav file")
    ap.add_argument("--self-test", action="store_true",
                    help="run built-in synthetic utterances (no wav needed)")
    ap.add_argument("--device", default="auto",
                    help="auto|cpu|cuda|mps (default auto; Pi5 -> cpu)")
    ap.add_argument("--max-frames", type=int, default=400,
                    help="max 10 ms frames (400 = 4 s); trimmed, not padded")
    ap.add_argument("--warmup", type=int, default=3,
                    help="warmup forward passes before timing (default 3)")
    ap.add_argument("--json", action="store_true", help="emit JSON on stdout")
    args = ap.parse_args(argv)

    device = get_device(args.device)
    ckpt = resolve_checkpoint(args.checkpoint)
    if not ckpt.exists():
        print(f"error: checkpoint not found: {ckpt}", file=sys.stderr)
        return 2

    model, intents, ctc_vocab, _ = load_model(str(ckpt), device)

    if args.input:
        wav = audio_utils.load_wav_mono(args.input)
        items = [(Path(args.input).name, wav)]
    elif args.self_test:
        items = self_test_wavs()
    else:
        print("error: provide --input <wav> or --self-test", file=sys.stderr)
        return 2

    # Warmup: absorbs first-call init / allocator cost so the reported
    # latency is steady-state (the number that matters for "instant").
    dummy = torch.zeros(1, 16, config.N_MELS, device=device)
    for _ in range(max(0, args.warmup)):
        model(dummy)

    results = []
    for name, wav in items:
        r = run_utterance(model, wav, device, args.max_frames,
                          intents, ctc_vocab)
        r["input"] = name
        results.append(r)

    if args.json:
        print(json.dumps({"device": str(device),
                          "checkpoint": str(ckpt),
                          "results": results}, indent=2))
    else:
        print(f"device={device}  checkpoint={ckpt}")
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
