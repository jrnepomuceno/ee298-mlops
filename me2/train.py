"""Training, validation, and testing loops for VCM."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from model import VCM, ctc_decode_batch, parse_slots
from utils import model_utils


def _loss_fn() -> nn.Module:
    return nn.CTCLoss(blank=config.CTC_BLANK, zero_infinity=True)


def _ctc_loss(ctc_loss: nn.CTCLoss,
              ctc_logits: torch.Tensor,
              ctc_targets: torch.Tensor,
              ctc_lengths: torch.Tensor,
              mels_lengths: torch.Tensor,
              intents: torch.Tensor) -> torch.Tensor:
    """CTC term with the two known-bug fixes applied:

    * OOV rows are masked out -- their transcripts tokenize to mostly
      ``<unk>`` and would otherwise poison the slot head (plan.md #2).
      The intent head still learns "this is oov" from the CE term.
    * ``input_lengths`` uses the TRUE utterance length (conv downsamples
      mel frames by 4), not the padded batch length (plan.md #4).
    """
    oov_id = config.INTENT_TO_ID[config.OOV_INTENT]
    keep = (intents != oov_id)
    n = intents.size(0)
    if not bool(keep.any()):
        # whole batch OOV: no CTC signal, keep the graph alive for backward
        return ctc_logits.sum() * 0.0
    kept_idx = keep.nonzero(as_tuple=True)[0]
    if not bool(keep.all()):
        # ctc_targets is flattened over the batch; re-slice it to the kept rows
        offsets = torch.cat([ctc_lengths.new_zeros(1), ctc_lengths.cumsum(0)])
        starts = offsets[kept_idx]
        ends = offsets[kept_idx + 1]
        ctc_targets = torch.cat([ctc_targets[s:e] for s, e in zip(starts, ends)])
        ctc_logits = ctc_logits[keep]
        ctc_lengths = ctc_lengths[keep]
        mels_lengths = mels_lengths[keep]
    input_lengths = torch.clamp(mels_lengths // 4, min=1)
    log_probs = nn.functional.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)
    ctc_kept = ctc_loss(log_probs, ctc_targets, input_lengths, ctc_lengths)
    # Scale to a per-sample mean over the WHOLE batch (OOV rows count as 0)
    # so the reported metric and gradient weight match the CE term, which
    # is averaged over all rows.
    return ctc_kept * kept_idx.numel() / n


def train_one_epoch(model: VCM,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    ctc_loss: nn.CTCLoss,
                    grad_clip: float = 5.0,
                    amp: bool = False) -> dict:
    model.train()
    total = 0.0
    ce_sum = 0.0
    ctc_sum = 0.0
    correct = 0
    n = 0
    t0 = time.time()
    # bf16 autocast for the forward+loss only; backward/step run in fp32
    # (bf16 needs no GradScaler). Enabled only on cuda.
    amp_enabled = bool(amp) and device.type == "cuda"

    for mels, intents, _transcripts, ctc_targets, ctc_lengths, mels_lengths in loader:
        mels = mels.to(device)
        intents = intents.to(device)
        ctc_targets = ctc_targets.to(device)
        ctc_lengths = ctc_lengths.to(device)
        mels_lengths = mels_lengths.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=amp_enabled):
            intent_logits, ctc_logits = model(mels, lengths=mels_lengths)

            # intent term
            ce = nn.functional.cross_entropy(intent_logits, intents)

            # ctc term (OOV-masked, true input lengths)
            ctc = _ctc_loss(ctc_loss, ctc_logits, ctc_targets, ctc_lengths,
                            mels_lengths, intents)

            loss = config.CE_WEIGHT * ce + config.CTC_WEIGHT * ctc
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total += loss.item() * intents.size(0)
        ce_sum += ce.item() * intents.size(0)
        ctc_sum += ctc.item() * intents.size(0)
        correct += (intent_logits.argmax(dim=-1) == intents).sum().item()
        n += intents.size(0)

    return {
        "loss": total / max(n, 1),
        "ce": ce_sum / max(n, 1),
        "ctc": ctc_sum / max(n, 1),
        "acc": correct / max(n, 1),
        "epoch_seconds": time.time() - t0,
    }


@torch.no_grad()
def validate(model: VCM,
             loader: DataLoader,
             device: torch.device,
             ctc_loss: nn.CTCLoss,
             max_decode: int = 4096) -> dict:
    """Validate: loss + intent metrics + CTC decode metrics (WER/CER/EM)."""
    model.eval()
    total = 0.0
    n = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    refs: list[str] = []
    hyps: list[str] = []

    for mels, intents, transcripts, ctc_targets, ctc_lengths, mels_lengths in loader:
        mels = mels.to(device)
        intents = intents.to(device)
        ctc_targets = ctc_targets.to(device)
        ctc_lengths = ctc_lengths.to(device)
        mels_lengths = mels_lengths.to(device)

        intent_logits, ctc_logits = model(mels, lengths=mels_lengths)
        ce = nn.functional.cross_entropy(intent_logits, intents)
        ctc = _ctc_loss(ctc_loss, ctc_logits, ctc_targets, ctc_lengths,
                        mels_lengths, intents)
        loss = config.CE_WEIGHT * ce + config.CTC_WEIGHT * ctc

        total += loss.item() * intents.size(0)
        n += intents.size(0)
        y_true.extend(intents.cpu().tolist())
        y_pred.extend(intent_logits.argmax(dim=-1).cpu().tolist())

        if len(hyps) < max_decode:
            refs.extend(transcripts)
            hyps.extend(ctc_decode_batch(ctc_logits.cpu()))

    intent_m = model_utils.intent_metrics(y_true, y_pred, config.INTENTS)
    ctc_m = model_utils.ctc_metrics(refs, hyps)
    return {"loss": total / max(n, 1), **intent_m, **ctc_m}


def test(model: VCM,
         loader: DataLoader,
         device: torch.device,
         ctc_loss: nn.CTCLoss,
         max_decode: int = 4096) -> dict:
    """Test: same as validate, plus per-sample examples for the report."""
    model.eval()
    total = 0.0
    n = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    refs: list[str] = []
    hyps: list[str] = []
    examples: list[dict] = []

    for mels, intents, transcripts, ctc_targets, ctc_lengths, mels_lengths in loader:
        mels = mels.to(device)
        intents = intents.to(device)
        ctc_targets = ctc_targets.to(device)
        ctc_lengths = ctc_lengths.to(device)
        mels_lengths = mels_lengths.to(device)

        intent_logits, ctc_logits = model(mels, lengths=mels_lengths)
        ce = nn.functional.cross_entropy(intent_logits, intents)
        ctc = _ctc_loss(ctc_loss, ctc_logits, ctc_targets, ctc_lengths,
                        mels_lengths, intents)
        loss = config.CE_WEIGHT * ce + config.CTC_WEIGHT * ctc

        total += loss.item() * intents.size(0)
        n += intents.size(0)
        preds = intent_logits.argmax(dim=-1).cpu().tolist()
        y_true.extend(intents.cpu().tolist())
        y_pred.extend(preds)
        if len(hyps) < max_decode:
            decoded = ctc_decode_batch(ctc_logits.cpu())
            refs.extend(transcripts)
            hyps.extend(decoded)
            for i in range(min(3, len(decoded))):
                intent = config.INTENTS[preds[i]]
                examples.append({
                    "intent": intent,
                    "intent_correct": bool(preds[i] == intents[i].item()),
                    "ref": transcripts[i],
                    "hyp": decoded[i],
                    "slots": parse_slots(intent, decoded[i]),
                })

    intent_m = model_utils.intent_metrics(y_true, y_pred, config.INTENTS)
    ctc_m = model_utils.ctc_metrics(refs, hyps)
    return {"loss": total / max(n, 1), **intent_m, **ctc_m,
            "examples": examples}
