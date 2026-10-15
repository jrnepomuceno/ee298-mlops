from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, RandomSampler

from pi5_vcm.dataset import NUM_INTENTS, SAMPLE_RATE, create_dataloader
from pi5_vcm.dataset_prepare import INTENT_NAMES
from pi5_vcm.dataset_source import DATASET_REVISION
from pi5_vcm.features import (
    FFT_SIZE,
    FRAME_HOP,
    FRAME_LENGTH,
    LogMelFrontend,
    compute_mel_statistics,
    save_mel_statistics,
)
from pi5_vcm.loss import inverse_frequency_class_weights, multitask_loss
from pi5_vcm.model import ME2Model, SLOT_NAMES, count_parameters

SEED = 42
STAGE_A_EPOCHS = 3
STAGE_B_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 3
STAGE_A_LEARNING_RATE = 1e-3
STAGE_B_ENCODER_LEARNING_RATE = 1e-4
STAGE_B_HEAD_LEARNING_RATE = 1e-3


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_reproducibility(seed: int = SEED, *, deterministic: bool = True) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = not deterministic


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid device {requested!r}; use auto, cpu, or cuda[:index]") from exc
    if device.type not in ("cpu", "cuda"):
        raise ValueError(f"invalid device type {device.type!r}; use cpu or cuda[:index]")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA device requested but CUDA is unavailable: {requested}")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {index} is unavailable; "
                f"found {torch.cuda.device_count()} device(s)"
            )
        return torch.device("cuda", index)
    return device


def resolve_precision(requested: str, device: torch.device) -> tuple[torch.dtype | None, str]:
    if requested not in ("auto", "fp32", "fp16", "bf16"):
        raise ValueError("precision must be auto, fp32, fp16, or bf16")
    if requested == "auto":
        if device.type == "cuda":
            return torch.float16, "fp16"
        return None, "fp32"
    if requested == "fp32":
        return None, "fp32"
    if device.type != "cuda":
        raise ValueError(f"{requested} mixed precision is supported only on CUDA")
    if requested == "bf16" and torch.cuda.get_device_capability(device)[0] < 8:
        raise ValueError(f"bfloat16 is not supported on {torch.cuda.get_device_name(device)}")
    return (torch.float16 if requested == "fp16" else torch.bfloat16), requested


def _hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_metadata(loader: DataLoader[dict[str, Any]]) -> dict[str, str]:
    dataset = loader.dataset
    while not hasattr(dataset, "manifest_path") and hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    manifest_path = getattr(dataset, "manifest_path", None)
    if manifest_path is None:
        return {"path": "<in-memory-dataset>", "sha256": "<unavailable>"}
    return {"path": str(manifest_path), "sha256": _hash_file(manifest_path)}


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_worktree_dirty() -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _run_metadata(
    *,
    train_loader: DataLoader[dict[str, Any]],
    validation_loader: DataLoader[dict[str, Any]],
    class_counts: list[int],
    device: torch.device,
    precision: str,
    deterministic: bool,
    seed: int,
    stage_a_epochs: int,
    stage_b_epochs: int,
    patience: int,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    source_paths = [
        Path(__file__).with_name(name)
        for name in (
            "dataset.py",
            "dataset_prepare.py",
            "dataset_source.py",
            "features.py",
            "loss.py",
            "model.py",
            "training.py",
        )
    ]
    source_paths.append(project_root / "training" / "run_metric_smoke.py")
    source_paths.extend(
        project_root / name for name in ("pyproject.toml", "uv.lock")
    )
    metadata: dict[str, Any] = {
        "seed": seed,
        "deterministic_algorithms": deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor() or platform.machine()
        ),
        "precision": precision,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "cuda_version": torch.version.cuda,
        "dataset_revision": DATASET_REVISION,
        "intent_names": list(INTENT_NAMES),
        "slot_names": list(SLOT_NAMES),
        "model_parameters": count_parameters(ME2Model()),
        "feature_config": {
            "sample_rate": SAMPLE_RATE,
            "mel_bins": 80,
            "frame_length_samples": FRAME_LENGTH,
            "frame_hop_samples": FRAME_HOP,
            "fft_size": FFT_SIZE,
        },
        "git_revision": _git_revision(),
        "git_worktree_dirty": _git_worktree_dirty(),
        "source_sha256": {
            str(path.relative_to(project_root)): _hash_file(path)
            for path in source_paths
        },
        "train_manifest": _manifest_metadata(train_loader),
        "validation_manifest": _manifest_metadata(validation_loader),
        "class_counts": class_counts,
        "training": {
            "stage_a_epochs": stage_a_epochs,
            "stage_a_learning_rate": STAGE_A_LEARNING_RATE,
            "stage_b_epochs": stage_b_epochs,
            "stage_b_encoder_learning_rate": STAGE_B_ENCODER_LEARNING_RATE,
            "stage_b_head_learning_rate": STAGE_B_HEAD_LEARNING_RATE,
            "early_stopping_patience": patience,
            "batch_size": train_loader.batch_size,
            "num_workers": train_loader.num_workers,
            "drop_last": train_loader.drop_last,
            "shuffle": isinstance(train_loader.sampler, RandomSampler),
        },
    }
    if device.type == "cuda":
        metadata["gpu_properties"] = {
            "name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        }
    return metadata


def macro_f1_score(targets: Tensor, predictions: Tensor) -> float:
    if targets.ndim != 1 or predictions.shape != targets.shape:
        raise ValueError("targets and predictions must be equally sized vectors")
    if targets.numel() == 0:
        raise ValueError("macro-F1 requires at least one target")
    if torch.any((targets < 0) | (targets >= NUM_INTENTS)).item():
        raise ValueError(f"targets must be in [0, {NUM_INTENTS - 1}]")
    if torch.any((predictions < 0) | (predictions >= NUM_INTENTS)).item():
        raise ValueError(f"predictions must be in [0, {NUM_INTENTS - 1}]")

    confusion = torch.bincount(
        targets * NUM_INTENTS + predictions, minlength=NUM_INTENTS**2
    ).reshape(NUM_INTENTS, NUM_INTENTS)
    support = confusion.sum(dim=1)
    true_positive = confusion.diagonal()
    predicted_count = confusion.sum(dim=0)
    denominator = support + predicted_count
    class_f1 = torch.where(
        denominator > 0,
        2.0 * true_positive / denominator.clamp_min(1),
        0.0,
    )
    represented = support > 0
    return class_f1[represented].mean().item()


def _batch_to_device(
    batch: dict[str, Any], device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    non_blocking = device.type == "cuda"
    waveforms = batch["audio"].to(
        device=device, dtype=torch.float32, non_blocking=non_blocking
    )
    audio_lengths = batch["audio_lengths"].to(device=device, non_blocking=non_blocking)
    intent_targets = batch["intent_targets"].to(
        device=device, dtype=torch.long, non_blocking=non_blocking
    )
    slot_targets = batch["slot_targets"].to(
        device=device, dtype=torch.long, non_blocking=non_blocking
    )
    slot_mask = batch["slot_mask"].to(
        device=device, dtype=torch.bool, non_blocking=non_blocking
    )
    return waveforms, audio_lengths, intent_targets, slot_targets, slot_mask


def _seed_data_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _run_epoch(
    model: ME2Model,
    frontend: LogMelFrontend,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    optimizer: Optimizer | None,
    class_weights: Tensor,
    precision_dtype: torch.dtype | None,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    all_targets: list[Tensor] = []
    all_predictions: list[Tensor] = []
    loss_totals = torch.zeros(3, dtype=torch.float64)
    example_count = 0

    for batch in loader:
        waveforms, lengths, targets, slots, slot_mask = _batch_to_device(batch, device)
        mel_features, frame_mask = frontend(waveforms, lengths)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type,
            dtype=precision_dtype or torch.float32,
            enabled=precision_dtype is not None,
        ):
            outputs = model(mel_features, frame_mask)
            losses = multitask_loss(
                outputs,
                targets,
                slots,
                slot_mask,
                intent_class_weights=class_weights,
            )
            if training:
                scaler.scale(losses["total"]).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = targets.shape[0]
        loss_totals += torch.tensor(
            [
                losses["total"].detach().item(),
                losses["intent"].detach().item(),
                losses["slots"].detach().item(),
            ],
            dtype=torch.float64,
        ) * batch_size
        example_count += batch_size
        all_targets.append(targets.detach().cpu())
        all_predictions.append(outputs["intent_logits"].argmax(dim=1).detach().cpu())

    if example_count == 0:
        raise ValueError("training and validation loaders must not be empty")
    targets_all = torch.cat(all_targets)
    predictions_all = torch.cat(all_predictions)
    average_losses = loss_totals / example_count
    return {
        "loss": average_losses[0].item(),
        "intent_loss": average_losses[1].item(),
        "slot_loss": average_losses[2].item(),
        "macro_f1": macro_f1_score(targets_all, predictions_all),
    }


def _atomic_torch_save(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as temporary_file:
            temp_path = temporary_file.name
        torch.save(value, temp_path)
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)


def train_model(
    train_loader: DataLoader[dict[str, Any]],
    validation_loader: DataLoader[dict[str, Any]],
    *,
    class_counts: list[int],
    output_dir: str | Path,
    device: torch.device | str = "cpu",
    stage_a_epochs: int = STAGE_A_EPOCHS,
    stage_b_epochs: int = STAGE_B_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    seed: int = SEED,
    precision: str = "auto",
    deterministic: bool = True,
) -> dict[str, Any]:
    if stage_a_epochs < 1 or stage_b_epochs < 1:
        raise ValueError("Stage A and Stage B must each run for at least one epoch")
    if patience < 1:
        raise ValueError("patience must be at least 1")
    configure_reproducibility(seed, deterministic=deterministic)
    device = resolve_device(str(device))
    precision_dtype, precision_name = resolve_precision(precision, device)
    for loader in (train_loader, validation_loader):
        if loader.generator is not None:
            loader.generator.manual_seed(seed)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    statistics_path = output_root / "mel_stats.npz"
    checkpoint_path = output_root / "me2_model.pt"
    model = ME2Model().to(device)
    frontend = LogMelFrontend().to(device)
    mean, std = compute_mel_statistics(frontend, train_loader, device=device)
    frontend.set_statistics(mean, std)
    save_mel_statistics(statistics_path, mean, std)

    class_weights = inverse_frequency_class_weights(class_counts).to(device)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision_dtype == torch.float16
    )
    run_metadata = _run_metadata(
        train_loader=train_loader,
        validation_loader=validation_loader,
        class_counts=class_counts,
        device=device,
        precision=precision_name,
        deterministic=deterministic,
        seed=seed,
        stage_a_epochs=stage_a_epochs,
        stage_b_epochs=stage_b_epochs,
        patience=patience,
    )
    run_metadata["mel_statistics_source"] = run_metadata["train_manifest"].copy()
    history: list[dict[str, Any]] = []

    model.set_encoder_trainable(False)
    stage_a_optimizer = Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=STAGE_A_LEARNING_RATE,
    )
    for epoch in range(1, stage_a_epochs + 1):
        train_metrics = _run_epoch(
            model,
            frontend,
            train_loader,
            device=device,
            optimizer=stage_a_optimizer,
            class_weights=class_weights,
            precision_dtype=precision_dtype,
            scaler=scaler,
        )
        validation_metrics = _run_epoch(
            model,
            frontend,
            validation_loader,
            device=device,
            optimizer=None,
            class_weights=class_weights,
            precision_dtype=precision_dtype,
            scaler=scaler,
        )
        history.append(
            {
                "stage": "A",
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )

    model.set_encoder_trainable(True)
    stage_b_optimizer = Adam(
        [
            {
                "params": model.encoder.parameters(),
                "lr": STAGE_B_ENCODER_LEARNING_RATE,
            },
            {
                "params": [
                    *model.intent_head.parameters(),
                    *(parameter for head in model.slot_heads for parameter in head.parameters()),
                ],
                "lr": STAGE_B_HEAD_LEARNING_RATE,
            },
        ]
    )
    best_macro_f1 = float("-inf")
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    best_epoch = 0
    for epoch in range(1, stage_b_epochs + 1):
        train_metrics = _run_epoch(
            model,
            frontend,
            train_loader,
            device=device,
            optimizer=stage_b_optimizer,
            class_weights=class_weights,
            precision_dtype=precision_dtype,
            scaler=scaler,
        )
        validation_metrics = _run_epoch(
            model,
            frontend,
            validation_loader,
            device=device,
            optimizer=None,
            class_weights=class_weights,
            precision_dtype=precision_dtype,
            scaler=scaler,
        )
        history.append(
            {
                "stage": "B",
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        score = validation_metrics["macro_f1"]
        if score > best_macro_f1:
            best_macro_f1 = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("Stage B completed without producing a best checkpoint")
    _atomic_torch_save(
        {
            "model_state_dict": best_state,
            "mel_mean": frontend.mean.detach().cpu(),
            "mel_std": frontend.std.detach().cpu(),
            "class_weights": class_weights.detach().cpu(),
            "seed": seed,
            "best_stage_b_epoch": best_epoch,
            "best_validation_macro_f1": best_macro_f1,
            "history": history,
            "run_metadata": run_metadata,
        },
        checkpoint_path,
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "mel_statistics": str(statistics_path),
        "best_stage_b_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "epochs_completed": len(history),
        "history": history,
        "run_metadata": run_metadata,
        "run_report": str(output_root / "training_run.json"),
    }
    report_path = Path(result["run_report"])
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _class_counts_from_loader(loader: DataLoader[dict[str, Any]]) -> list[int]:
    dataset = loader.dataset
    records = getattr(dataset, "records", None)
    if records is None:
        raise ValueError("training dataset must expose prepared manifest records")
    counts = Counter(record.intent_id for record in records)
    if None in counts:
        raise ValueError("training manifest contains an unlabeled record")
    return [counts[index] for index in range(NUM_INTENTS)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the ME2 model in the planned two supervised stages."
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="data loader workers (increase to overlap Parquet loading on Linux)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto selects CUDA when available, otherwise CPU; also accepts cpu or cuda[:index]",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
        help="auto uses CUDA fp16 with gradient scaling, or CPU fp32",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="allow faster nondeterministic kernels; deterministic training is the default",
    )
    arguments = parser.parse_args()
    configure_reproducibility(
        arguments.seed, deterministic=not arguments.non_deterministic
    )
    device = resolve_device(arguments.device)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(arguments.seed)

    train_loader = create_dataloader(
        arguments.train_manifest,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=arguments.num_workers,
        dataset_dir=arguments.dataset_dir,
        generator=loader_generator,
        pin_memory=device.type == "cuda",
        worker_init_fn=_seed_data_worker,
    )
    validation_loader = create_dataloader(
        arguments.validation_manifest,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        dataset_dir=arguments.dataset_dir,
        pin_memory=device.type == "cuda",
        worker_init_fn=_seed_data_worker,
    )
    result = train_model(
        train_loader,
        validation_loader,
        class_counts=_class_counts_from_loader(train_loader),
        output_dir=arguments.output_dir,
        device=device,
        precision=arguments.precision,
        seed=arguments.seed,
        deterministic=not arguments.non_deterministic,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
