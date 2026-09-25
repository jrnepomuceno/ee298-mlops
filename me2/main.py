"""Entry point for Pi5-VCM.

Usage:
    python main.py generate            # build synthetic manifest -> data/manifest.jsonl
    python main.py train --epochs 10   # train on the configured VCM manifest
    python main.py train --manifest data/real_manifest.jsonl   # alternate manifest
    python main.py test                # evaluate best checkpoint on test split
    python main.py demo                # run a few sample utterances through the model

Everything runs CPU-only on the Pi; device=auto picks cuda/mps if present.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import json
import random
import sys
import time
from pathlib import Path

import torch

import config
from dataset import (VCMDataset, build_synthetic_manifest, load_manifest,
                     manifest_fingerprint, split_samples)
from dataloader import make_dataloader
from model import VCM, ctc_decode_batch, parse_slots
from train import test, train_one_epoch, validate
from utils import audio_utils
from utils.model_utils import (get_device, load_checkpoint, save_checkpoint,
                               set_seed)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
DEFAULT_DATASET_ROOT = Path(r"C:\Users\jdrne\TrainingGround\datasets\voice_dataset")
DEFAULT_MANIFEST = (DEFAULT_DATASET_ROOT / "distilled" / "manifests" /
                    "manifest_vcm.jsonl")
BEST_CHECKPOINT = "pi5-vcm-best.pt"
LAST_CHECKPOINT = "pi5-vcm-last.pt"
HISTORY_FILE = "pi5-vcm-history.json"
MODEL_PRESETS = {
    "baseline": {"conv_channels": 64, "hidden_size": 128, "num_layers": 2},
    "large": {"conv_channels": 96, "hidden_size": 192, "num_layers": 2},
}


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        text = "\n".join(
            f"[{timestamp}] {line}" if line else line
            for line in text.split("\n")
        )
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pi5-VCM command model")
    p.add_argument("command", choices=["generate", "train", "test", "demo"],
                   help="generate: make synthetic manifest; train: fit model; "
                        "test: evaluate; demo: run sample utterances")
    p.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST),
                   help="path to real-data manifest (CSV/JSONL). "
                        f"(default: {DEFAULT_MANIFEST})")
    p.add_argument("--mels-dir", type=str, default=None,
                   help="directory containing precomputed mels and sidecars")
    p.add_argument("--num-samples", type=int, default=2000,
                   help="synthetic samples per split source (default 2000)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--resume", type=str, default=None,
                   help="resume training from a checkpoint")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-frames", type=int, default=400,
                   help="max mel frames per utterance (400 = 4s @ 10ms hop)")
    p.add_argument("--num-workers", type=int, default=0,
                   help="keep 0 on the Pi; 4-8 on a dev machine")
    p.add_argument("--pin-memory", action="store_true",
                   help="pin CPU batches for async H2D copies (GPU runs)")
    p.add_argument("--noise-snr", type=float, default=None,
                   help="train-only: add white Gaussian noise to the log-mel "
                        "at a random SNR in [value, value+10] dB (e.g. 15)")
    p.add_argument("--amp", action="store_true",
                   help="bfloat16 autocast for the training step (cuda only)")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt", type=str, default=None,
                   help=f"checkpoint path for test/demo (default: {BEST_CHECKPOINT})")
    p.add_argument("--model-size", choices=MODEL_PRESETS, default="baseline",
                   help="model capacity preset; large increases weights and RAM")
    p.add_argument("--conv-channels", type=int, default=None)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--log-file", type=str, default=None,
                   help="tee training output to this file")
    p.add_argument("--output-dir", type=str, default=None,
                   help="directory for training checkpoints and history")
    return p.parse_args()


def get_split_loaders(args, device) -> tuple:
    """Build train/val/test datasets + dataloaders from args."""
    manifest_path = Path(args.manifest)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {manifest_path}. "
            "Pass --manifest with a CSV/JSONL manifest path."
        )

    print(f"[data] loading manifest {manifest_path}")
    samples = load_manifest(manifest_path)

    train_s, val_s, test_s = split_samples(samples, seed=args.seed)
    print(f"[data] train={len(train_s)} val={len(val_s)} test={len(test_s)}")
    mels_fingerprint = manifest_fingerprint(samples) if args.mels_dir else None

    train_ds = VCMDataset(train_s, max_frames=args.max_frames, augment=True,
                          mels_dir=args.mels_dir, split="train",
                          manifest_seed=args.seed,
                          mels_fingerprint=mels_fingerprint,
                          noise_snr=args.noise_snr)
    val_ds = VCMDataset(val_s, max_frames=args.max_frames, augment=False,
                        mels_dir=args.mels_dir, split="val",
                        manifest_seed=args.seed,
                        mels_fingerprint=mels_fingerprint)
    test_ds = VCMDataset(test_s, max_frames=args.max_frames, augment=False,
                         mels_dir=args.mels_dir, split="test",
                         manifest_seed=args.seed,
                         mels_fingerprint=mels_fingerprint)

    train_loader = make_dataloader(train_ds, args.batch_size, shuffle=True,
                                   num_workers=args.num_workers,
                                   pin_memory=args.pin_memory)
    val_loader = make_dataloader(val_ds, args.batch_size, shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=args.pin_memory)
    test_loader = make_dataloader(test_ds, args.batch_size, shuffle=False,
                                  num_workers=args.num_workers,
                                  pin_memory=args.pin_memory)
    return train_loader, val_loader, test_loader, test_ds


def build_model(args, device) -> VCM:
    preset = MODEL_PRESETS[args.model_size]
    model = VCM(conv_channels=args.conv_channels or preset["conv_channels"],
                hidden_size=args.hidden_size or preset["hidden_size"],
                num_layers=args.num_layers or preset["num_layers"],
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
    training_started = time.perf_counter()
    device = get_device(args.device)
    set_seed(args.seed)
    print(f"[train] device={device}")
    output_dir = Path(args.output_dir) if args.output_dir else CKPT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, _test_ds = get_split_loaders(args, device)
    model = build_model(args, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)
    ctc_loss = torch.nn.CTCLoss(blank=config.CTC_BLANK, zero_infinity=True)

    start_epoch = 0
    best_val = float("inf")
    history: list[dict] = []
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        model, payload = load_checkpoint(resume_path, device, model=model,
                                         optimizer=optimizer)
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr
        start_epoch = int(payload.get("epoch", 0))
        history_path = output_dir / HISTORY_FILE
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        best_val = min((row.get("val_loss", float("inf")) for row in history),
                       default=float("inf"))
        print(f"[resume] checkpoint={resume_path} start_epoch={start_epoch} "
              f"best_val_loss={best_val:.4f}")

    end_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch + 1, end_epoch + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, device,
                                  ctc_loss, amp=args.amp)
        val_m = validate(model, val_loader, device, ctc_loss)
        scheduler.step(val_m["loss"])

        history.append({"epoch": epoch, **train_m,
                        "val_loss": val_m["loss"], "val_acc": val_m["accuracy"],
                        "val_macro_f1": val_m["macro_f1"],
                        "val_wer": val_m["wer"]})

        print(f"[epoch {epoch:02d}/{end_epoch}] "
              f"train_loss={train_m['loss']:.4f} train_acc={train_m['acc']:.4f} | "
              f"val_loss={val_m['loss']:.4f} val_acc={val_m['accuracy']:.4f} "
              f"macro_f1={val_m['macro_f1']:.4f} wer={val_m['wer']:.4f} | "
              f"{time.time() - t0:.1f}s")

        if val_m["loss"] < best_val:
            best_val = val_m["loss"]
            save_checkpoint(output_dir / BEST_CHECKPOINT, model, optimizer, epoch,
                            {"val": val_m, "train": train_m}, args)
            print(f"[ckpt] saved {BEST_CHECKPOINT} (val_loss={best_val:.4f})")

    save_checkpoint(output_dir / LAST_CHECKPOINT, model, optimizer, end_epoch,
                    {"val": val_m, "train": train_m}, args)
    with open(output_dir / HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[train] done. history -> {output_dir / HISTORY_FILE}")

    test_metrics = test(model, test_loader, device, ctc_loss)
    test_metrics.pop("examples", None)
    total_seconds = time.perf_counter() - training_started
    print("\n===== TRAINING REPORT =====")
    print(f"train_loss  : {train_m['loss']:.4f}")
    print(f"train_acc   : {train_m['acc']:.4f}")
    print(f"val_loss    : {val_m['loss']:.4f}")
    print(f"val_acc     : {val_m['accuracy']:.4f}")
    print(f"val_macro_f1: {val_m['macro_f1']:.4f}")
    print(f"val_wer     : {val_m['wer']:.4f}")
    print(f"test_loss   : {test_metrics['loss']:.4f}")
    print(f"test_acc    : {test_metrics['accuracy']:.4f}")
    print(f"test_macro_f1: {test_metrics['macro_f1']:.4f}")
    print(f"test_wer    : {test_metrics['wer']:.4f}")
    print(f"test_cer    : {test_metrics['cer']:.4f}")
    print(f"test_exact  : {test_metrics['exact_match']:.4f}")
    print(f"total_seconds: {total_seconds:.2f}")


def _load_model_for_eval(args, device) -> VCM:
    ckpt = Path(args.ckpt) if args.ckpt else CKPT_DIR / BEST_CHECKPOINT
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
    if args.command == "train" and args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8", buffering=1) as log:
            with redirect_stdout(_Tee(sys.stdout, log)), \
                    redirect_stderr(_Tee(sys.stderr, log)):
                cmd_train(args)
        return
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
