"""Precompute variable-length fbank features without truncating utterances."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from new_training.data import FEATURE_BINS, load_package, split_fingerprint
from new_training.audio_features import load_wav_mono, mel_spectrogram


def _compute_mel(item: tuple[str, int | None, str]) -> np.ndarray:
    audio_path, max_frames, dtype_name = item
    mel = mel_spectrogram(load_wav_mono(audio_path))
    if max_frames is not None:
        mel = mel[:max_frames]
    return np.asarray(mel.numpy(), dtype=np.dtype(dtype_name), order="C")


def _metadata_matches(out_dir: Path, split: str, fingerprint: str,
                     n_rows: int, dtype_name: str, max_frames: int | None) -> bool:
    metadata_path = out_dir / f"features_{split}.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        lengths = np.load(out_dir / metadata["lengths_file"], allow_pickle=False)
        offsets = np.load(out_dir / metadata["offsets_file"], allow_pickle=False)
        feature_path = out_dir / metadata["feature_file"]
        dtype = np.dtype(dtype_name)
        return (
            metadata.get("manifest_fingerprint") == fingerprint and
            metadata.get("n_rows") == n_rows and
            metadata.get("dtype") == dtype_name and
            metadata.get("max_frames") == max_frames and
            metadata.get("n_mels") == FEATURE_BINS and
            lengths.shape == offsets.shape == (n_rows,) and
            feature_path.is_file() and
            feature_path.stat().st_size == int(lengths.sum()) * FEATURE_BINS * dtype.itemsize
        )
    except (OSError, KeyError, ValueError, TypeError):
        return False


def _precompute_split(rows: list[dict], split: str, out_dir: Path,
                      workers: int, dtype_name: str,
                      max_frames: int | None) -> None:
    fingerprint = split_fingerprint(rows)
    if _metadata_matches(out_dir, split, fingerprint, len(rows), dtype_name, max_frames):
        print(f"[{split}] current; skipping", flush=True)
        return

    feature_name = f"features_{split}.{dtype_name}"
    offsets_name = f"offsets_{split}.npy"
    lengths_name = f"lengths_{split}.npy"
    feature_tmp = out_dir / f".{feature_name}.tmp"
    offsets_tmp = out_dir / f".{offsets_name}.tmp"
    lengths_tmp = out_dir / f".{lengths_name}.tmp"
    metadata_tmp = out_dir / f".features_{split}.json.tmp"
    offsets: list[int] = []
    lengths: list[int] = []
    frame_offset = 0
    started = time.perf_counter()
    inputs = ((row["path"], max_frames, dtype_name) for row in rows)

    try:
        with feature_tmp.open("wb") as feature_stream:
            if workers > 0:
                with mp.Pool(processes=workers) as pool:
                    feature_iter = pool.imap(_compute_mel, inputs, chunksize=8)
                    for index, mel in enumerate(feature_iter, 1):
                        offsets.append(frame_offset)
                        lengths.append(int(mel.shape[0]))
                        mel.tofile(feature_stream)
                        frame_offset += int(mel.shape[0])
                        if index % 1000 == 0:
                            print(f"[{split}] {index}/{len(rows)}", flush=True)
            else:
                for index, item in enumerate(inputs, 1):
                    mel = _compute_mel(item)
                    offsets.append(frame_offset)
                    lengths.append(int(mel.shape[0]))
                    mel.tofile(feature_stream)
                    frame_offset += int(mel.shape[0])
                    if index % 1000 == 0:
                        print(f"[{split}] {index}/{len(rows)}", flush=True)
            feature_stream.flush()
            os.fsync(feature_stream.fileno())

        with offsets_tmp.open("wb") as stream:
            np.save(stream, np.asarray(offsets, dtype=np.int64), allow_pickle=False)
        with lengths_tmp.open("wb") as stream:
            np.save(stream, np.asarray(lengths, dtype=np.int64), allow_pickle=False)

        metadata = {
            "format_version": 1,
            "split": split,
            "manifest_fingerprint": fingerprint,
            "n_rows": len(rows),
            "total_frames": frame_offset,
            "n_mels": FEATURE_BINS,
            "dtype": dtype_name,
            "max_frames": max_frames,
            "feature_file": feature_name,
            "offsets_file": offsets_name,
            "lengths_file": lengths_name,
        }
        metadata_tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        feature_tmp.replace(out_dir / feature_name)
        offsets_tmp.replace(out_dir / offsets_name)
        lengths_tmp.replace(out_dir / lengths_name)
        metadata_tmp.replace(out_dir / f"features_{split}.json")
    finally:
        for temp_path in (feature_tmp, offsets_tmp, lengths_tmp, metadata_tmp):
            if temp_path.exists():
                temp_path.unlink()

    print(f"[{split}] wrote {len(rows)} rows / {frame_offset:,} frames in "
          f"{time.perf_counter() - started:.1f}s", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True,
                        help="training_package_capped directory")
    parser.add_argument("--out-dir", required=True,
                        help="directory for variable-length feature stores")
    parser.add_argument("--workers", type=int, default=4,
                        help="precompute processes; use 0 for single-process smoke tests")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="optional explicit truncation cap; default keeps every frame")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 0:
        raise ValueError("workers must be >= 0")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max-frames must be >= 1")
    labels, splits = load_package(args.package_root)
    print(f"[data] labels={len(labels)} rows="
          f"{sum(map(len, splits.values()))}", flush=True)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        _precompute_split(rows, split, out_dir, args.workers, args.dtype,
                          args.max_frames)


if __name__ == "__main__":
    main()
