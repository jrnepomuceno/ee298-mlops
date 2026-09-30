"""Small real-audio smoke test; temporary feature files are removed on exit."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from new_training.data import MelFeatureDataset, load_package, make_loader
from new_training.model import IntentModel
from new_training.precompute_mels import _precompute_split
from new_training.train_intent import _run_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True)
    package_root = Path(parser.parse_args().package_root)
    labels, splits = load_package(package_root)
    sample_rows = {
        split: [rows[0]]
        for split, rows in (("train", splits["train"]),
                            ("val", splits["val"]),
                            ("test", splits["test"]))
    }

    with tempfile.TemporaryDirectory(prefix="vcm-smoke-", dir=Path(__file__).parent) as temp:
        features_dir = Path(temp)
        for split, rows in sample_rows.items():
            _precompute_split(rows, split, features_dir, workers=0,
                              dtype_name="float16", max_frames=None)
        train_ds = MelFeatureDataset(sample_rows["train"], labels, features_dir,
                                     "train", augment=True)
        val_ds = MelFeatureDataset(sample_rows["val"], labels, features_dir, "val")
        test_ds = MelFeatureDataset(sample_rows["test"], labels, features_dir, "test")
        train_loader = make_loader(train_ds, 1, True, 0, False)
        val_loader = make_loader(val_ds, 1, False, 0, False)
        test_loader = make_loader(test_ds, 1, False, 0, False)

        model = IntentModel(num_intents=len(labels))
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        class_weights = torch.ones(len(labels))
        try:
            train_metrics = _run_epoch(model, train_loader, optimizer,
                                       torch.device("cpu"), class_weights,
                                       amp=False, max_batches=1, training=True)
            val_metrics = _run_epoch(model, val_loader, optimizer,
                                     torch.device("cpu"), class_weights,
                                     amp=False, max_batches=1, training=False)
            test_metrics = _run_epoch(model, test_loader, optimizer,
                                      torch.device("cpu"), class_weights,
                                      amp=False, max_batches=1, training=False)
            assert train_metrics["examples"] == val_metrics["examples"] == test_metrics["examples"] == 1
        finally:
            train_ds.close()
            val_ds.close()
            test_ds.close()
        print("[smoke] package -> precompute -> mmap -> bucketed loader -> train/eval passed")


if __name__ == "__main__":
    main()
