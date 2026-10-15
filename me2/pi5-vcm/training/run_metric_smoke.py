from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from pi5_vcm.dataset import ParquetAudioDataset, collate_audio
from pi5_vcm.model import NUM_INTENTS
from pi5_vcm.training import (
    SEED,
    configure_reproducibility,
    resolve_device,
    train_model,
)


def _select_balanced_indices(
    records: list[Any], *, examples_per_class: int, seed: int
) -> list[int]:
    indices_by_class: dict[int, list[int]] = {class_id: [] for class_id in range(NUM_INTENTS)}
    for index, record in enumerate(records):
        if record.intent_id in indices_by_class:
            indices_by_class[record.intent_id].append(index)
    missing = [
        class_id
        for class_id, indices in indices_by_class.items()
        if len(indices) < examples_per_class
    ]
    if missing:
        raise ValueError(
            f"Not enough examples for intents {missing}; "
            f"need {examples_per_class} per class"
        )

    rng = random.Random(seed)
    selected: list[int] = []
    for class_id in range(NUM_INTENTS):
        class_indices = indices_by_class[class_id]
        selected.extend(rng.sample(class_indices, examples_per_class))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small, stratified GPU training smoke test and save a report; "
            "this is not a full training run."
        )
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("training/reports"))
    parser.add_argument("--examples-per-class", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--seed", type=int, default=SEED)
    arguments = parser.parse_args()
    if arguments.examples_per_class < 1:
        raise ValueError("--examples-per-class must be at least 1")
    if arguments.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    configure_reproducibility(arguments.seed)
    device = resolve_device(arguments.device)
    train_dataset = ParquetAudioDataset(
        arguments.train_manifest, dataset_dir=arguments.dataset_dir
    )
    validation_dataset = ParquetAudioDataset(
        arguments.validation_manifest, dataset_dir=arguments.dataset_dir
    )
    train_indices = _select_balanced_indices(
        train_dataset.records,
        examples_per_class=arguments.examples_per_class,
        seed=arguments.seed,
    )
    validation_indices = _select_balanced_indices(
        validation_dataset.records,
        examples_per_class=arguments.examples_per_class,
        seed=arguments.seed + 1,
    )
    train_generator = torch.Generator().manual_seed(arguments.seed)
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=arguments.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_audio,
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_audio,
    )
    class_counts = [
        sum(record.intent_id == class_id for record in train_dataset.records)
        for class_id in range(NUM_INTENTS)
    ]

    output_dir = arguments.report_dir / "3060ti-smoke-artifacts"
    started = time.perf_counter()
    result = train_model(
        train_loader,
        validation_loader,
        class_counts=class_counts,
        output_dir=output_dir,
        device=device,
        stage_a_epochs=1,
        stage_b_epochs=1,
        patience=1,
        seed=arguments.seed,
        precision=arguments.precision,
        deterministic=True,
    )
    elapsed_seconds = time.perf_counter() - started

    report_lines = [
        "# 3060 Ti training metric smoke test",
        "",
        "**Purpose:** verify the pinned data-to-model training path and collect",
        "initial optimization metrics. This tiny stratified run is not a full",
        "training run and its validation Macro-F1 is not a model-quality estimate.",
        "",
        "## Run configuration",
        "",
        f"- Device: {result['run_metadata']['device_name']} ({result['run_metadata']['device']})",
        f"- Precision: {result['run_metadata']['precision']}",
        f"- Seed: {arguments.seed}; deterministic algorithms enabled",
        f"- Training subset: {len(train_indices)} examples "
        f"({arguments.examples_per_class} per intent)",
        f"- Validation subset: {len(validation_indices)} examples "
        f"({arguments.examples_per_class} per intent)",
        f"- Batch size: {arguments.batch_size}",
        "- Training budget: one Stage A epoch and one Stage B epoch",
        f"- Elapsed time (including mel-stat fitting): {elapsed_seconds:.2f} seconds",
        f"- Checkpoint: `{result['checkpoint']}`",
        f"- Full run metadata: `{result['run_report']}`",
        "",
        "## Metrics",
        "",
        "| Stage | Epoch | Train loss | Train intent loss | Train slot loss | Train Macro-F1 | Validation loss | Validation intent loss | Validation slot loss | Validation Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for epoch in result["history"]:
        train = epoch["train"]
        validation = epoch["validation"]
        report_lines.append(
            f"| {epoch['stage']} | {epoch['epoch']} | {train['loss']:.5f} | "
            f"{train['intent_loss']:.5f} | {train['slot_loss']:.5f} | "
            f"{train['macro_f1']:.5f} | {validation['loss']:.5f} | "
            f"{validation['intent_loss']:.5f} | {validation['slot_loss']:.5f} | "
            f"{validation['macro_f1']:.5f} |"
        )
    report_lines.extend(
        [
            "",
            "Metrics are from fixed, small, class-balanced subsets. Use the full",
            "training and validation manifests for a real fit and meaningful",
            "validation estimates; this smoke run intentionally does not do that.",
            "",
        ]
    )
    arguments.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = arguments.report_dir / "3060ti-metric-smoke.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(report_path)
    print(result["run_report"])
    print(f"subset sizes: train={len(train_indices)}, validation={len(validation_indices)}")
    print(f"elapsed_seconds: {elapsed_seconds:.2f}")
    for epoch in result["history"]:
        print(
            f"stage={epoch['stage']} epoch={epoch['epoch']} "
            f"train_loss={epoch['train']['loss']:.5f} "
            f"train_macro_f1={epoch['train']['macro_f1']:.5f} "
            f"validation_loss={epoch['validation']['loss']:.5f} "
            f"validation_macro_f1={epoch['validation']['macro_f1']:.5f}"
        )


if __name__ == "__main__":
    main()
