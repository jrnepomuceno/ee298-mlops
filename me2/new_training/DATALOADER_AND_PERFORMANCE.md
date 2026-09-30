# Dataloader and 3060 Ti Performance Plan

Target: Windows training host with an NVIDIA RTX 3060 Ti (8 GB VRAM). The code path is implemented under `new_training/` and has passed a real-audio CPU smoke test. Full feature precompute/training has not yet run; the dataset archive is checksum-verified on Daniel-PC. No Pi code or deployment artifact has been changed.

The requested final artifact is an INT8 ONNX model, but that is not produced by data loading. The loader feeds training and produces batches; training produces a PyTorch checkpoint. Export and quantization are a separate downstream stage, covered under [INT8 ONNX Handoff](#int8-onnx-handoff).

## Existing Loader

`model/dataloader.py::collate_fn` pads each batch to its longest feature sequence, creates the two classification/CTC targets, and preserves true mel-frame lengths. `make_dataloader` supports worker processes, pinned host memory, and persistent workers. `model/dataset.py::VCMDataset` can read precomputed `.npy` features through a read-only mmap and opens the mmap inside each worker, which is appropriate for Windows' spawn-based workers.

There is one padding inefficiency to address in the new training path: `VCMDataset.__getitem__` pads each example to `max_frames` before collation. With `max_frames=400`, the collator therefore receives fixed-length rows and cannot reduce padding for short utterances. Keep fixed-size rows on disk if convenient, but trim each loaded row to its saved true length before returning it; let batch collation pad only to that batch's maximum. Combine this with duration-aware bucketed sampling so nearby lengths share batches. Preserve the true lengths through the convolution's x4 temporal downsampling for both masked intent pooling and CTC loss.

## Recommended Input Pipeline

1. First normalize the three package manifests in this new training track: resolve package-relative `audio_path`, rename it to `path`, map `intent_label` to an explicit new label ID, and normalize `validation` to `val`. Preserve the package's existing split membership. Do not call random splitting on the source data.
2. The new precompute defaults to preserving all frames; `--max-frames` is optional and explicit. The source has 2,596 examples longer than 4 seconds, so do not set a cap without measuring the effect on task labels. True lengths and the chosen cap are recorded and validated.
3. Precompute deterministic log-mels once using `new_training/audio_features.py`, which was checked bit-identical to the existing fbank implementation on representative real clips (16 kHz, Kaldi fbank, 80 bins, 25 ms window, 10 ms shift, zero dither). Keep augmentation out of this deterministic step. Retain train-only SpecAugment after loading features.
4. Store each split as contiguous variable-length FP16 feature rows in a flat file, with offsets, lengths, and a metadata sidecar. This avoids padding the entire corpus to 400 frames and keeps host storage/page-cache traffic lower. The loader mmaps features and converts only requested rows to tensors; it does not preload the feature corpus into GPU memory.
5. Load from a local SSD/NVMe when possible. Start with 4 Windows DataLoader workers, pinned memory, persistent workers, and prefetching (where supported/configured). Benchmark 2/4/6 workers; more workers are not automatically faster, especially when augmentation or storage becomes the bottleneck.
6. Transfer only the current, length-bucketed minibatch to the GPU. Use pinned batches with `.to("cuda", non_blocking=True)`; avoid preloading the entire feature corpus into the GPU. Fixed 400-frame FP16 features alone occupy roughly 4.73 GiB, leaving insufficient reliable room on an 8 GB card for model activations, gradients, optimizer state, and workspaces.

## 3060 Ti Training Order

1. Establish a short baseline with precomputed mels, dynamic batch padding, and a fixed seed. Record examples/second, epoch time, GPU utilization, peak allocated/reserved VRAM, CPU utilization, and data-wait time.
2. Enable CUDA BF16 autocast (`--amp` in the current trainer) and TF32 matmul/cuDNN where enabled by the existing entry point. The 3060 Ti is Ampere-class; BF16 avoids GradScaler management. Compare validation metrics and throughput.
3. Start at batch size 32. Try 64 after measuring peak memory, keeping several hundred MiB of headroom for CUDA/cuDNN workspaces and validation. Increase or reduce from measured throughput/OOM behavior rather than assuming a batch-size target. Dynamic padding can improve the usable batch size.
4. Keep `pin_memory=True`, non-blocking transfer, persistent workers, and tune worker count/prefetch together. Measure with GPU utilization: if it is frequently idle between steps, improve the input path; if it is consistently busy, worker increases are unlikely to help.
5. Only then benchmark a fused AdamW optimizer and `torch.compile`. `torch.compile` has startup and graph-shape costs; length bucketing can limit shape variation, but compilation is not a first-line fix for an I/O-bound run. Benchmark compilation after warm-up and include compile time separately.
6. Consider channels-last only if profiling shows the convolution stack benefits and conversions do not erase the gain. The input and convolution layout transition must be measured end-to-end. The model has a GRU and no attention block, so FlashAttention/SDPA recommendations do not apply.
7. Profile one representative epoch before architecture changes. Avoid per-step `.item()`, `.cpu()`, or frequent logging; the existing trainer accumulates training metrics on-device and synchronizes at epoch end.

## GPU Memory and Data Movement

The current baseline is a small parameter model, but activation memory scales with batch size and padded time-frequency dimensions. BF16 autocast reduces activation and bandwidth pressure; dynamic padding and length buckets reduce the number of feature values transferred and processed. Pinned host memory can overlap host-to-device copies with compute, while precomputed features remove repeated WAV decode and fbank work from every epoch.

“GPU SRAM” is not an explicitly managed PyTorch data store: registers/shared memory/cache are managed by CUDA kernels. At the application level, the practical controls are reducing host-to-device bytes, avoiding unnecessary layout/cast copies, keeping intermediates in fused kernels when beneficial, and keeping the GPU fed. Do not optimize for theoretical SRAM movement without a profiler trace; track wall-clock throughput and utilization.

The existing precompute path converts mmap rows to float32 tensors in `VCMDataset`, so float16 files reduce storage and disk/page-cache traffic but do not by themselves halve H2D traffic. After correctness is established, benchmark a batch path that retains lower-precision host batches and transfers them directly under autocast. Confirm model input compatibility, feature fidelity, and validation metrics before changing the default.

## Required Correctness Gates

- Confirm normalized train/validation/test row counts and label counts match the package files exactly; verify no sample ID or speaker crosses source partitions unexpectedly.
- Confirm precompute row `i`, length `i`, and normalized record `i` refer to the same utterance in all splits. Reject stale/mismatched fingerprints.
- Compare precomputed features with on-the-fly features for representative short, near-cap, and long clips.
- Measure transcript tokenizer coverage and CTC-feasible lengths before training. The current vocabulary misses about 29.7% of package transcript tokens; do not hide this with `<unk>` metrics.
- Verify no validation/test augmentation, and apply noise only to training data.
- After the adapter/model contract is approved, run a small one-batch forward/backward smoke test and one short validation pass on the Windows CUDA environment before a full run.

## Implemented First Slice

The new training code is intent-only with the package's 18 labels in `intents.json` order. It does not invent an OOV class or train the existing CTC head, which currently maps about 29.7% of transcript tokens to `<unk>`. Slot learning can follow separately using `slot_values` or a revised transcript tokenizer after coverage and target feasibility are measured. Do not create a compatibility mapping to the existing Pi intents without a reviewed semantic mapping; checkpoints and ONNX outputs remain separate from the deployed model.

## INT8 ONNX Handoff

This follows training; it is not part of the dataloader:

1. Select and validate the best PyTorch checkpoint against the new training taxonomy and its explicit intent ordering/CTC vocabulary.
2. Export that exact architecture to FP32 ONNX with the expected `(B, T, 80)` float32 mel input and named outputs. The existing exporter emits `intent_logits` and `ctc_logits`; a changed head or intent-only model requires a matching exporter and output contract.
3. Check ONNX numerical parity against PyTorch on representative short and long inputs before quantization.
4. Optimize the graph if appropriate, then run the repository's ONNX Runtime dynamic quantizer. Its current path quantizes eligible weights to INT8 while activations remain FP32; it does not require calibration data and applies a file-size budget check.
5. Compare FP32 ONNX and INT8 ONNX logits and task metrics on held-out validation data; check file size and desktop CPU latency. The quantizer's optional benchmark is not Pi validation.
6. Stop before transfer or device deployment in this task. Any later deployment requires a separate approval and remote/device validation.

An INT8 ONNX file from this new dataset is not automatically compatible with the current Pi application. Label IDs, output dimensions, CTC vocabulary, feature extraction, and downstream intent/slot interpretation must agree end to end. Quantization changes numerical representation, not those semantics. Keep this inference-contract gate distinct from loader throughput tuning.
