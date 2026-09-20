"""Checkpointing, metrics, and small shared helpers."""

from __future__ import annotations

import numpy as np
import torch

import config


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device(device: str = "auto") -> torch.device:
    """auto -> cuda if available, else mps (Apple Silicon), else cpu."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path, model, optimizer, epoch, metrics, args=None) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metrics": metrics,
        "config": {
            "num_mels": config.N_MELS,
            "num_intents": config.NUM_INTENTS,
            "ctc_vocab_size": config.CTC_VOCAB_SIZE,
            "intents": config.INTENTS,
            "ctc_vocab": config.CTC_VOCAB,
            "args": vars(args) if args is not None else None,
        },
    }
    torch.save(payload, str(path))


def load_checkpoint(path, device, model=None, optimizer=None):
    """Load a checkpoint; build a fresh VCM if `model` is None."""
    payload = torch.load(str(path), map_location=device)
    if model is None:
        from model import VCM
        model = VCM()
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return model, payload


# ------------------------------------------------------------------ metrics --

def intent_metrics(y_true, y_pred, intent_names) -> dict:
    """Accuracy, macro-F1, and per-intent P/R/F1 (pure python, no sklearn)."""
    n = len(y_true)
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    per_intent = {}
    f1s = []
    for i, name in enumerate(intent_names):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == i and p == i)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != i and p == i)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == i and p != i)
        support = sum(1 for t in y_true if t == i)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        per_intent[name] = {"precision": round(precision, 4),
                            "recall": round(recall, 4),
                            "f1": round(f1, 4),
                            "support": support}
        if support:
            f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return {"accuracy": correct / max(n, 1), "macro_f1": macro_f1,
            "per_intent": per_intent}


def _levenshtein(a: list, b: list) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


def word_error_rate(ref: str, hyp: str) -> float:
    """WER with the same tokenization as the CTC head (digits per char)."""
    r, h = config.transcript_to_tokens(ref), hyp.split()
    return _levenshtein(r, h) / max(len(r), 1)


def character_error_rate(ref: str, hyp: str) -> float:
    r = "".join(config.transcript_to_tokens(ref))
    h = "".join(hyp.split())
    return _levenshtein(list(r), list(h)) / max(len(r), 1)


def ctc_metrics(refs, hyps) -> dict:
    """WER / CER / exact-match between decoded text and transcripts."""
    if not refs:
        return {"wer": 0.0, "cer": 0.0, "exact_match": 0.0}
    wer = float(np.mean([word_error_rate(r, h) for r, h in zip(refs, hyps)]))
    cer = float(np.mean([character_error_rate(r, h) for r, h in zip(refs, hyps)]))
    em = float(np.mean([int(config.transcript_to_tokens(r) == h.split())
                        for r, h in zip(refs, hyps)]))
    return {"wer": wer, "cer": cer, "exact_match": em}
