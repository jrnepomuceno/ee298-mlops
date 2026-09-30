"""Train the capped package's 18-class intent-only baseline."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from new_training.data import MelFeatureDataset, load_package, make_loader
from new_training.model import IntentModel


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(confusion: torch.Tensor, loss_sum: torch.Tensor,
             count: int) -> dict[str, float]:
    diagonal = confusion.diag().float()
    support = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()
    f1 = 2 * diagonal / (support + predicted).clamp_min(1)
    return {
        "loss": float((loss_sum / max(count, 1)).item()),
        "accuracy": float((diagonal.sum() / max(count, 1)).item()),
        "macro_f1": float(f1.mean().item()),
    }


def _run_epoch(model: IntentModel, loader, optimizer, device: torch.device,
               class_weights: torch.Tensor, amp: bool, max_batches: int | None,
               training: bool) -> dict[str, float]:
    model.train(training)
    classes = model.num_intents
    confusion = torch.zeros((classes, classes), dtype=torch.int64, device=device)
    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    count = 0
    started = time.perf_counter()
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, (mels, labels, lengths) in enumerate(loader):
            mels = mels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            if device.type != "cuda" or not amp:
                mels = mels.float()
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=amp and device.type == "cuda"):
                logits = model(mels, lengths)
                loss = F.cross_entropy(logits, labels, weight=class_weights)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            batch_size = labels.size(0)
            loss_sum += loss.detach().float() * batch_size
            predictions = logits.detach().argmax(dim=1)
            bins = labels * classes + predictions
            confusion += torch.bincount(bins, minlength=classes * classes).reshape(classes, classes)
            count += batch_size
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    metrics = _metrics(confusion, loss_sum, count)
    metrics["examples"] = count
    metrics["seconds"] = time.perf_counter() - started
    return metrics


def _save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--output-dir", default="new_training/runs/intent_v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="limit each split for a post-precompute smoke run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs/batch-size must be positive and num-workers non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    _seed_everything(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"[train] device={device} amp={device.type == 'cuda' and not args.no_amp}", flush=True)

    labels, rows = load_package(args.package_root)
    features_dir = Path(args.features_dir)
    train_ds = MelFeatureDataset(rows["train"], labels, features_dir, "train", augment=True)
    val_ds = MelFeatureDataset(rows["val"], labels, features_dir, "val")
    test_ds = MelFeatureDataset(rows["test"], labels, features_dir, "test")
    pin_memory = device.type == "cuda"
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers,
                               pin_memory, args.seed)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers,
                             pin_memory, args.seed)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers,
                              pin_memory, args.seed)

    counts = np.bincount(train_ds.label_ids, minlength=len(labels)).astype(np.float32)
    class_weights = torch.from_numpy(counts.sum() / (len(labels) * counts)).to(device)
    model = IntentModel(num_intents=len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    use_amp = device.type == "cuda" and not args.no_amp

    for epoch in range(args.epochs):
        train_loader.batch_sampler.set_epoch(epoch)
        train_metrics = _run_epoch(model, train_loader, optimizer, device,
                                   class_weights, use_amp, args.max_batches, True)
        val_metrics = _run_epoch(model, val_loader, optimizer, device,
                                 class_weights, use_amp, args.max_batches, False)
        record = {"epoch": epoch + 1, "train": train_metrics, "validation": val_metrics}
        history.append(record)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"[epoch {epoch + 1:03d}] "
              f"train_loss={train_metrics['loss']:.4f} "
              f"train_f1={train_metrics['macro_f1']:.4f} "
              f"val_loss={val_metrics['loss']:.4f} "
              f"val_macro_f1={val_metrics['macro_f1']:.4f} "
              f"train_s={train_metrics['seconds']:.1f}", flush=True)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            stale_epochs = 0
            _save_checkpoint(output_dir / "best.pt", {
                "model_state_dict": model.state_dict(),
                "model_config": {"n_mels": 80, "num_intents": len(labels),
                                 "conv_channels": 64, "hidden_size": 128,
                                 "num_layers": 2, "dropout": 0.3},
                "labels": labels,
                "feature_config": {"sample_rate": 16000, "n_mels": 80,
                                    "frame_length_ms": 25.0, "frame_shift_ms": 10.0,
                                    "dtype": "float16 storage; cast per batch"},
                "task": "intent_classification_only",
                "best_epoch": best_epoch,
                "best_validation_macro_f1": best_f1,
            })
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"[train] early stop at epoch {epoch + 1}; best epoch={best_epoch}", flush=True)
            break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = _run_epoch(model, test_loader, optimizer, device, class_weights,
                              use_amp, args.max_batches, False)
    checkpoint["test_metrics"] = test_metrics
    _save_checkpoint(output_dir / "best.pt", checkpoint)
    print(f"[test] loss={test_metrics['loss']:.4f} "
          f"accuracy={test_metrics['accuracy']:.4f} "
          f"macro_f1={test_metrics['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
