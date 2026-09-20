"""Entry point for Pi5-VCM.

Usage:
    python main.py generate            # build synthetic manifest -> data/manifest.jsonl
    python main.py train --epochs 10   # train on synthetic data (default)
    python main.py train --manifest data/real_manifest.jsonl   # real data
    python main.py test                # evaluate best checkpoint on test split
    python main.py demo                # run a few sample utterances through the model

Everything runs CPU-only on the Pi; device=auto picks cuda/mps if present.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

import config
from dataset import (VCMDataset, build_synthetic_manifest, load_manifest,
                     split_samples)
from dataloader import make_dataloader
from model import VCM, ctc_decode_batch, parse_slots
from train import test, train_one_epoch, validate
from utils import audio_utils
from utils.model_utils import get_device, save_checkpoint, set_seed

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pi5-VCM command model")
    p.add_argument("command", choices=["generate", "train", "test", "demo"],
                   help="generate: make synthetic manifest; train: fit model; "
                        "test: evaluate; demo: run sample utterances")
    p.add_argument("--manifest", type=str, default=None,
                   help="path to real-data manifest (CSV/JSONL). "
                        "If omitted, synthetic data is generated.")
    p.add_argument("--num-samples", type=int, default=2000,
                   help="synthetic samples per split source (default 2000)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-frames", type=int, default=400,
                   help="max mel frames per utterance (400 = 4s @ 10ms hop)")
    p.add_argument("--num-workers", type=int, default=0,
                   help="keep 0 on the Pi")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt", type=str, default=None,
                   help="checkpoint path for test/demo (default: best.pt)")
    p.add_argument("--conv-channels", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    return p.parse_args()


def get_split_loaders(args, device) -> tuple:
    """Build train/val/test datasets + dataloaders from args."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else DATA_DIR / "manifest.jsonl"

    if not manifest_path.exists():
        print(f"[data] generating synthetic manifest -> {manifest_path}")
        samples = build_synthetic_manifest(args.num_samples, seed=args.seed)
        with open(manifest_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
    else:
        print(f"[data] loading manifest {manifest_path}")
        samples = load_manifest(manifest_path)

    train_s, val_s, test_s = split_samples(samples, seed=args.seed)
    print(f"[data] train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    train_ds = VCMDataset(train_s, max_frames=args.max_frames, augment=True)
    val_ds = VCMDataset(val_s, max_frames=args.max_frames, augment=False)
    test_ds = VCMDataset(test_s, max_frames=args.max_frames, augment=False)

    train_loader = make_dataloader(train_ds, args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)
    val_loader = make_dataloader(val_ds, args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)
    test_loader = make_dataloader(test_ds, args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)
    return train_loader, val_loader, test_loader, test_ds


def build_model(args, device) -> VCM:
    model = VCM(conv_channels=args.conv_channels,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout)
    model.to(device)
    n_params = model.count_parameters()
    print(f"[model] VCM params={n_params:,} "
          f"~{n_params * 4 / 1e6:.2f} MB float32, "
          f"~{n_params / 1e6:.2f} MB int8")
    return model


def cmd_generate(args) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    samples = build_synthetic_manifest(args.num_samples, seed=args.seed)
    out = DATA_DIR / "manifest.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"[generate] wrote {len(samples)} samples -> {out}")


def cmd_train(args) -> None:
    device = get_device(args.device)
    set_seed(args.seed)
    print(f"[train] device={device}")

    train_loader, val_loader, _test_loader, _test_ds = get_split_loaders(args, device)
    model = build_model(args, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)
    ctc_loss = torch.nn.CTCLoss(blank=config.CTC_BLANK, zero_infinity=True)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, device, ctc_loss)
        val_m = validate(model, val_loader, device, ctc_loss)
        scheduler.step(val_m["loss"])

        history.append({"epoch": epoch, **train_m,
                        "val_loss": val_m["loss"], "val_acc": val_m["accuracy"],
                        "val_macro_f1": val_m["macro_f1"],
                        "val_wer": val_m["wer"]})

        print(f"[epoch {epoch:02d}/{args.epochs}] "
              f"train_loss={train_m['loss']:.4f} train_acc={train_m['acc']:.4f} | "
              f"val_loss={val_m['loss']:.4f} val_acc={val_m['accuracy']:.4f} "
              f"macro_f1={val_m['macro_f1']:.4f} wer={val_m['wer']:.4f} | "
              f"{time.time() - t0:.1f}s")

        if val_m["loss"] < best_val:
            best_val = val_m["loss"]
            save_checkpoint(CKPT_DIR / "best.pt", model, optimizer, epoch,
                            {"val": val_m, "train": train_m}, args)
            print(f"[ckpt] saved best.pt (val_loss={best_val:.4f})")

    save_checkpoint(CKPT_DIR / "last.pt", model, optimizer, args.epochs,
                    {"val": val_m, "train": train_m}, args)
    with open(CKPT_DIR / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[train] done. history -> {CKPT_DIR / 'history.json'}")


def _load_model_for_eval(args, device) -> VCM:
    ckpt = Path(args.ckpt) if args.ckpt else CKPT_DIR / "best.pt"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt} (run train first)")
    model, _payload = torch_load_checkpoint(ckpt, device)
    return model


def torch_load_checkpoint(ckpt, device):
    from utils.model_utils import load_checkpoint
    return load_checkpoint(ckpt, device)


def cmd_test(args) -> None:
    device = get_device(args.device)
    set_seed(args.seed)
    _tl, _vl, test_loader, _test_ds = get_split_loaders(args, device)
    model = _load_model_for_eval(args, device)
    ctc_loss = torch.nn.CTCLoss(blank=config.CTC_BLANK, zero_infinity=True)

    metrics = test(model, test_loader, device, ctc_loss)
    examples = metrics.pop("examples", [])

    print("\n===== TEST RESULTS =====")
    print(f"loss        : {metrics['loss']:.4f}")
    print(f"accuracy    : {metrics['accuracy']:.4f}")
    print(f"macro_f1    : {metrics['macro_f1']:.4f}")
    print(f"WER         : {metrics['wer']:.4f}")
    print(f"CER         : {metrics['cer']:.4f}")
    print(f"exact_match : {metrics['exact_match']:.4f}")
    print("\nper-intent:")
    for name, m in metrics["per_intent"].items():
        print(f"  {name:<18} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} n={m['support']}")
    print("\nsample decodes:")
    for ex in examples:
        mark = "ok " if ex["intent_correct"] else "ERR"
        print(f"  [{mark}] {ex['intent']:<18} ref='{ex['ref']}' hyp='{ex['hyp']}' "
              f"slots={ex['slots']}")


def cmd_demo(args) -> None:
    """Run a handful of sample utterances through the model and print
    intent + decoded transcript + parsed slots."""
    device = get_device(args.device)
    model = _load_model_for_eval(args, device)

    demo_lines = [
        ("turn on the lights",),
        ("dim the lights to 40 percent",),
        ("set the temperature to 22 degrees",),
        ("set a timer for 2 minutes",),
        ("set an alarm for 7 am",),
        ("remind me to water the plants",),
        ("call mom",),
        ("what time is it",),
        ("what are my reminders",),
        ("tell me a joke",),
    ]
    print(f"\n===== DEMO (device={device}) =====")
    for (line,) in demo_lines:
        words = line.split()
        wav = audio_utils.synthesize_utterance(words, speaker="demo", seed=7)
        mel = audio_utils.mel_spectrogram(torch.from_numpy(wav))
        mel = audio_utils.pad_or_trim(mel, args.max_frames).unsqueeze(0).to(device)

        t0 = time.time()
        with torch.no_grad():
            intent_logits, ctc_logits = model(mel)
        intent = config.INTENTS[intent_logits.argmax(dim=-1).item()]
        hyp = ctc_decode_batch(ctc_logits.cpu())[0]
        slots = parse_slots(intent, hyp)
        ms = (time.time() - t0) * 1000

        print(f"  audio='{line}'")
        print(f"    intent={intent:<18} transcript='{hyp}' slots={slots} "
              f"infer={ms:.0f}ms")


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "demo":
        cmd_demo(args)


if __name__ == "__main__":
    main()
