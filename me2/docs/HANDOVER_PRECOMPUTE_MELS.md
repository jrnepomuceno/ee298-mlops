# Handover — Precomputed-Mels Training Pipeline
**Date:** 2026-09-24 · **Status:** DESIGN APPROVED, NOT YET IMPLEMENTED
**Task:** implement the precompute-mels feature (3 files), verify, and sync to daniel-pc.

## 1. Why
Training is I/O-bound: the data pipeline re-reads and re-encodes all 48,538 WAV/FLAC files
every epoch (on-the-fly fbank in `dataset.py:_mel()`). Empirical ceiling on daniel-pc:
**~120 s/epoch** (batch 128, 8 workers, AMP on). Baseline without the speed flags was 139 s.
The model itself is trivial (1.96 MB); the disk is the bottleneck. Precomputing mels moves
the bottleneck to GPU compute; expected **~20–40 s/epoch** (3–5×). This makes the planned
bigger-model retrain (hidden 192, 3 layers) affordable.

## 2. Design (approved by user 2026-09-24)
Split the per-sample pipeline at the stochastic boundary:

```
load_wav_mono → fbank(80,25ms,10ms,dither=0) → pad_or_trim(400)   ← DETERMINISTIC: precompute once
spec_augment (train mode only)                                     ← STOCHASTIC: stays in __getitem__
```

- **Storage:** one `.npy` per split — `mels_train.npy` (38,532 rows), `mels_val.npy` (5,381),
  `mels_test.npy` (4,625). Shape `(N, 400, 80)`, float32, C-contiguous. Total 5.78 GB fp32
  (2.89 GB fp16 fallback). Loaded with `np.load(..., mmap_mode='r')` — the OS page cache makes
  epoch 2+ RAM-resident on daniel-pc (16 GB+ RAM). All DataLoader workers share the page cache.
- **Sidecar JSON** next to each npy: `{seed, manifest_fingerprint, n_rows, max_frames, n_mels, dtype}`.
  **manifest_fingerprint must be path-agnostic** (hash of record content excluding the `path`
  field, e.g. sha256 over sorted `intent|split|transcript` per row) — the Mac manifest has Mac
  absolute paths, daniel-pc has Windows absolute paths; the same logical manifest must validate
  on both.
- **Row alignment:** the precompute script calls the SAME `split_samples(samples, seed=42)`
  that `main.py` calls. Row `i` of `mels_<split>.npy` must be the mel of the sample
  `VCMDataset` returns at index `i`.
- **All 48,538 records are real audio (zero synthetic)** — the precompute path is uniform.
  The dataset must still refuse synthetic samples (no `path`) on the mels path with a clear error.

## 3. Implementation (3 files, in `me2_work/` first, then sync)

### 3.1 `me2/precompute_mels.py` (NEW, ~80 lines)
- Args: `--manifest`, `--out-dir`, `--seed 42`, `--workers 8`, `--dtype float32|float16`.
- Load manifest → `split_samples(samples, seed=args.seed)` (import from `dataset.py`, do NOT
  reimplement) → for each split, process records in order with a process pool:
  exact `_mel()` sequence with `augment=False` (import `load_wav_mono`, `mel_spectrogram`,
  `pad_or_trim` from `utils/audio_utils.py` — no reimplementation; fbank params live there).
- Write `mels_<split>.npy` + `mels_<split>.json` sidecar per split.
- Resume-safe: skip a split if npy exists and sidecar matches (seed + fingerprint + n_rows).
- **Fail loudly on any missing audio file** — doubles as the path-existence gate.
- Progress log every 1,000 files + per-split wall time.

### 3.2 `me2/dataset.py` (~15 lines changed)
- `VCMDataset.__init__(..., mels_dir=None)`. If given:
  - `self._mels = np.load(Path(mels_dir) / f"mels_{split}.npy", mmap_mode='r')`
  - Verify `n_rows == len(samples)` AND sidecar seed/fingerprint match — **raise on any
    mismatch** (a silent row/label misalignment would train on shuffled labels with no error).
- `_mel()` short-circuit: `mel = torch.from_numpy(np.ascontiguousarray(self._mels[idx]))`
  (fresh copy — the mmap is never mutated), then the existing `spec_augment` / `pad_or_trim`
  lines run **unchanged**.
- fp16 files: `.float()` after load (negligible copy).

### 3.3 `me2/main.py` (2 lines)
- `--mels-dir` arg, passed through to the three dataset constructors in `get_split_loaders`.

## 4. Validation protocol (run in order; stop on any failure)
1. **Bit-identity:** precompute 100 samples per split; compare each row against the on-the-fly
   `_mel(augment=False)` output with `torch.equal`. Same functions + same inputs ⇒ must be
   identical. If not, something drifted (fbank params, pad order) — stop.
2. **One-epoch A/B on daniel-pc:** one full epoch with and without `--mels-dir`, same seed,
   same worker count. Same sample order + same RNG stream ⇒ SpecAugment draws identical masks ⇒
   **val_loss must match to ~1e-6**. Proves training dynamics are unchanged, not just features.
3. Normal run with `--mels-dir`; watch `epoch_seconds` on the per-epoch log line.

## 5. Sync to daniel-pc (after local verification)
- **SSH (verified working 2026-09-24):**
  `ssh -o BatchMode=yes -i ~/.ssh/id_ed25519_jdrne_daniel_pc jdrne@daniel-pc.tailfa8657.ts.net 'powershell -NoProfile -Command "..."'`
  ⚠️ `~/.ssh/config` has a `daniel-pc` alias but its `IdentityFile`
  (`~/.ssh/id_ed25519_danielpc`) is STALE — BatchMode auth fails with it. Use the explicit
  `-i ~/.ssh/id_ed25519_jdrne_daniel_pc` key. (Optionally fix the config entry.)
- **Remote layout (verified):**
  - Training code: `C:\Users\jdrne\TrainingGround\v0.1.0-training\`
    (main.py, dataset.py, dataloader.py, train.py, config.py, model.py, eval_slots.py,
    decode_probe.py, `checkpoints/`)
  - Manifest: `C:\Users\jdrne\TrainingGround\datasets\voice_dataset\distilled\manifests\manifest_vcm.jsonl`
    (48,538 records, Windows absolute paths, the combined 3-dataset manifest)
  - Python: `C:\Users\jdrne\AppData\Local\Programs\Python\Python312` (torch 2.6.0+cu124,
    soundfile installed — audio backend works)
- **Sync steps:** scp the 3 changed/new files into `v0.1.0-training\` → run
  `python precompute_mels.py --manifest <manifest_vcm.jsonl> --out-dir C:\Users\jdrne\TrainingGround\mels --seed 42 --workers 8`
  (one-time, ~2–4 min) → retrain with `--mels-dir C:\Users\jdrne\TrainingGround\mels`.
- **Mac caveat:** `--num-workers > 0` CRASHES on macOS (spawn pickles VCMDataset whose
  `_cache` holds MPS tensors). Use `--num-workers 0` on the Mac, `--num-workers 8` on daniel-pc.
  The A/B validation (step 2) must run on daniel-pc, not the Mac.

## 6. State at handover (verified 2026-09-24)
- **Model:** `pi5-vcm-best.pt` = fine-tuned epoch-65 weights (val_macro_f1 0.9536,
  test_macro_f1 0.9643). On daniel-pc: `v0.1.0-training\checkpoints\pi5-vcm-last.pt`
  (23,542,275 B, 9/23 20:20) — the best checkpoint is the final-epoch weights.
- **Digit-join: DISCARDED everywhere (user decision 2026-09-24).** Repo, daniel-pc, and
  `me2_work/` all run the original `parse_slots()` (`t = transcript.lower()`, no `re.sub`).
  Do NOT re-apply it. Consequence: deployed numeric slot extraction is the "before fix"
  behavior (18→8, 10→0); the ~97% end-to-end slot numbers from the eval do NOT apply to
  the deployed parser. The only surviving copy of the join logic is inside
  `me2_work/eval_slots.py` (session artifact, left for reproducibility).
- **Repo** (`~/MyPlayground/Pi5-VCM`): last commit `adac071` (2026-09-22). Uncommitted:
  training-speed edits (main.py, dataloader.py, train.py — `--pin-memory`, `--amp`,
  persistent_workers), README/docs/pipeline.sh tweaks, deleted `reply_catalog.py` + 2 dist
  tarballs, untracked `best.pt.synthetic.bak` + `recording/speakers.txt`. User's call to commit.
- **Docs trilogy** in `~/MyPlayground/Datasets/`: `DATASETS.md` (inventory),
  `distilled/FIX_PLAN.md` (manifest→VCM contract), `INTENT_COVERAGE.md` (16-intent coverage,
  gaps, ranked optimizations: record cancel_timer, distill MLEnd as slot track, cap kws
  single-words, per-intent caps).
- **`me2_work/`** = working copy of `me2/` (session dir). Implement here, then sync.

## 7. Open decisions (not blocking implementation)
1. fp32 (default, 5.78 GB) vs fp16 (2.89 GB) — fp32 fits daniel-pc; fp16 only if RAM is tight.
2. Whether to also precompute for the Pi (OptionB-only manifest, 17,996 files → 2.3 GB fp32,
   fits in 8 GB RAM) — only if Pi epoch time ever matters.
3. Committing the uncommitted repo changes — user's call.
