For PyTorch training on an 8 GB card, the same three levels apply, but most of the gains come from a handful of settings.

**1. Host → device (data loading)**
```python
loader = DataLoader(ds, batch_size=64, shuffle=True,
                    num_workers=4, pin_memory=True,
                    persistent_workers=True, prefetch_factor=2)

for x, y in loader:
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
```
- `pin_memory=True` plus `non_blocking=True` lets the copy overlap with compute.
- If the dataset is small enough (a few GB), preload it onto the GPU once and skip the loader entirely.
- Do augmentation on the GPU where possible (e.g. torchvision v2 transforms on tensors, Kornia) so you move small raw data instead of large processed data.
- Avoid `.item()`, `.cpu()`, and `print(loss)` every step. Each one forces a sync and a device → host copy. Accumulate on GPU and log every N steps.

**2. Cut bytes moved and stored on the GPU**
- **Mixed precision** is the biggest single win. Ampere has good FP16/BF16 tensor core support:
```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(x).loss
```
Use BF16 if you want to skip the `GradScaler`, or FP16 with `torch.cuda.amp.GradScaler()`. This roughly halves activation memory and traffic.
- **TF32 for remaining FP32 matmuls:** `torch.backends.cuda.matmul.allow_tf32 = True` (or `torch.set_float32_matmul_precision("high")`).
- **Activation checkpointing** (`torch.utils.checkpoint`) trades extra compute for much lower activation memory. This is often what lets you fit a larger batch or model in 8 GB.
- **Gradient accumulation** gives a larger effective batch without the memory cost of a larger real batch.
- **Fused optimizers:** `torch.optim.AdamW(params, fused=True)` does the update in one kernel instead of many, cutting memory round trips. Also `zero_grad(set_to_none=True)`.
- **8-bit optimizers** (bitsandbytes) shrink Adam's state, which is 2x the parameter count in FP32.

**3. Fewer trips to global memory inside the model**
- **`torch.compile(model)`** fuses elementwise ops (norm, activation, bias, dropout) so intermediates stay in registers rather than round-tripping to VRAM. It's usually a free speedup for transformer and CNN training.
- **`F.scaled_dot_product_attention`** dispatches to FlashAttention or memory-efficient kernels, avoiding materializing the n×n attention matrix. That's the O(n²) memory issue from your architecture table.
- **`channels_last`** memory format for CNNs (`model.to(memory_format=torch.channels_last)`) works better with tensor cores under AMP.

**4. Diagnose before tuning**
- `torch.cuda.memory_summary()` and `torch.cuda.max_memory_allocated()` show your actual peak.
- `torch.profiler` (with TensorBoard or Chrome trace) shows whether you're waiting on the dataloader, on copies, or on kernels. If the GPU sits idle between steps, the bottleneck is input, not the model.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` if you hit fragmentation-related OOMs despite having free memory.

**Rough order to try:** AMP (BF16) → pinned memory and workers → `torch.compile` → fused AdamW → SDPA attention → checkpointing if still out of memory.

If you share your model type and batch size, I can suggest which of these will matter most, or help read a profiler trace.