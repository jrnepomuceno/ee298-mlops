"""Precompute deterministic log-mel features for a real-data manifest."""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np
import torch

from dataset import load_manifest, manifest_fingerprint, split_samples
from utils.audio_utils import load_wav_mono, mel_spectrogram, pad_or_trim


MAX_FRAMES = 400


def _compute(sample: dict) -> tuple[np.ndarray, int]:
    """(padded mel, true pre-padding frame count) for one sample."""
    if "path" not in sample:
        raise ValueError("precomputed mels require real samples with paths")
    path = Path(sample["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    mel = mel_spectrogram(load_wav_mono(str(path)))
    true_len = min(int(mel.shape[0]), MAX_FRAMES)
    return (pad_or_trim(mel, MAX_FRAMES).numpy().astype(np.float32, copy=False),
            true_len)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_manifest(args.manifest)
    splits = split_samples(samples, seed=args.seed)
    fingerprint = manifest_fingerprint(samples)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_samples_list in zip(("train", "val", "test"), splits):
        npy_path = out_dir / f"mels_{split_name}.npy"
        sidecar_path = npy_path.with_suffix(".json")
        if npy_path.exists() and sidecar_path.exists():
            with open(sidecar_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            if (metadata.get("seed") == args.seed and
                    metadata.get("manifest_fingerprint") == fingerprint and
                    metadata.get("n_rows") == len(split_samples_list)):
                print(f"[{split_name}] up-to-date; skipping")
                continue
        started = time.perf_counter()
        rows = []
        lengths = []
        with Pool(processes=args.workers) as pool:
            for index, (row, true_len) in enumerate(pool.imap(_compute, split_samples_list), 1):
                rows.append(row)
                lengths.append(true_len)
                if index % 1000 == 0:
                    print(f"[{split_name}] {index}/{len(split_samples_list)}", flush=True)
        array = np.ascontiguousarray(np.stack(rows))
        if args.dtype == "float16":
            array = array.astype(np.float16)
        np.save(npy_path, array)
        lengths_path = npy_path.with_name(f"mels_{split_name}_lengths.npy")
        np.save(lengths_path, np.asarray(lengths, dtype=np.int64))
        metadata = {
            "seed": args.seed,
            "manifest_fingerprint": fingerprint,
            "n_rows": len(split_samples_list),
            "max_frames": MAX_FRAMES,
            "n_mels": 80,
            "dtype": str(array.dtype),
            "lengths_file": lengths_path.name,
        }
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"[{split_name}] wrote {npy_path} in "
              f"{time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()