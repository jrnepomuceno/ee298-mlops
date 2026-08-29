"""
train.py — cluster-ready training loop for the einops/einsum MNIST CNN in model.py.

Single GPU / CPU:
    python train.py --epochs 3

One node, 8x A100:
    torchrun --nproc_per_node=8 train.py --amp --amp-dtype bf16 --batch-size 256

Multi-node (e.g. Slurm, 4 nodes x 8 GPUs):
    torchrun --nnodes=4 --node_rank=$SLURM_NODEID --nproc_per_node=8 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        train.py --amp --amp-dtype bf16 --batch-size 256

Notes:
    --batch-size is PER-GPU. Effective global batch = batch-size x world_size.
    Use --scale-lr linear to scale LR by world_size (linear scaling rule).
    --subset N caps training samples for quick smoke tests on the cluster.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from model import MNISTCNN


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the einsum MNIST CNN (DDP + AMP ready)")

    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128,
                   help="per-GPU batch size (global = batch-size x world_size)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--scale-lr", choices=["none", "linear"], default="none",
                   help="linear: multiply LR by world_size")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=0.0,
                   help="max gradient norm; 0 disables clipping")
    p.add_argument("--seed", type=int, default=0)

    # Mixed precision (recommended on A100: bf16 needs no loss scaling)
    p.add_argument("--amp", action="store_true",
                   help="enable automatic mixed precision")
    p.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16",
                   help="bf16 for A100/Ampere+; fp16 falls back to loss scaling")

    # Data
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--subset", type=int, default=0,
                   help="cap training-set size for smoke tests (0 = full)")

    # Distributed (RANK/WORLD_SIZE/LOCAL_RANK are picked up from torchrun)
    p.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    p.add_argument("--dist-url", type=str, default=None,
                   help="init method; default env:// (set by torchrun)")

    # Misc
    p.add_argument("--log-interval", type=int, default=50,
                   help="log every N steps (0 disables step logging)")
    p.add_argument("--checkpoint", type=str, default="mnist_cnn_einsum_best.pt")
    p.add_argument("--save-last", action="store_true",
                   help="also save a final-epoch checkpoint")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Distributed helpers
# --------------------------------------------------------------------------- #
def distributed_info() -> tuple[int, int, int, bool]:
    """(rank, world_size, local_rank, is_distributed) from torchrun env vars."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def pick_device(local_rank: int, is_dist: bool) -> torch.device:
    if torch.cuda.is_available():
        if is_dist:
            torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def make_loaders(args, is_dist: bool) -> tuple[DataLoader, DataLoader, int]:
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=tfm)

    if args.subset > 0:
        n = min(args.subset, len(train_set))
        train_set = Subset(train_set, list(range(n)))

    train_sampler = DistributedSampler(train_set, shuffle=True) if is_dist else None
    test_sampler = DistributedSampler(test_set, shuffle=False) if is_dist else None

    use_cuda = torch.cuda.is_available()
    common = dict(
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=is_dist,  # keep per-rank batch sizes equal for DDP
        **common,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        **common,
    )
    return train_loader, test_loader, len(train_set)


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_enabled, amp_dtype) -> tuple[float, float]:
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        seen += x.size(0)
    return total_loss / seen, 100.0 * correct / seen


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    args,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    rank: int,
) -> float:
    model.train()
    total_loss, seen = 0.0, 0
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled):
            loss = criterion(model(x), y)

        if amp_enabled and amp_dtype == torch.float16:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        seen += x.size(0)
        if rank == 0 and args.log_interval > 0 and (step + 1) % args.log_interval == 0:
            print(f"    step {step + 1:4d}/{len(loader)}  loss {total_loss / seen:.4f}",
                  flush=True)
    return total_loss / seen


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, is_dist = distributed_info()
    torch.manual_seed(args.seed + rank)  # per-rank seed; samplers handle sharding

    if is_dist:
        dist.init_process_group(
            backend=args.backend,
            init_method=args.dist_url or "env://",
        )

    device = pick_device(local_rank, is_dist)
    is_main = rank == 0

    if is_main:
        print(f"torch={torch.__version__}  device={device}  "
              f"world_size={world_size}  amp={'on (' + args.amp_dtype + ')' if args.amp else 'off'}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    model = MNISTCNN().to(device)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"MNISTCNN: {n_params:,} parameters")

    if is_dist:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    train_loader, test_loader, n_train = make_loaders(args, is_dist)
    if is_main:
        print(f"train samples: {n_train}  per-gpu batch: {args.batch_size}  "
              f"global batch: {args.batch_size * world_size}")

    criterion = nn.CrossEntropyLoss()
    lr = args.lr * world_size if args.scale_lr == "linear" else args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    # bf16 needs no loss scaling; fp16 does.
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and amp_dtype == torch.float16)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, args, args.amp, amp_dtype, rank,
        )
        val_loss, val_acc = evaluate(
            model, test_loader, criterion, device, args.amp, amp_dtype
        )
        dt = time.time() - t0
        if is_main:
            print(f"epoch {epoch}/{args.epochs}  train_loss {train_loss:.4f}  "
                  f"val_loss {val_loss:.4f}  val_acc {val_acc:.2f}%  ({dt:.1f}s)",
                  flush=True)
            if val_acc > best_acc:
                best_acc = val_acc
                obj = model.module if hasattr(model, "module") else model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": obj.state_dict(),
                        "val_acc": val_acc,
                        "args": vars(args),
                    },
                    args.checkpoint,
                )
                print(f"  -> saved best checkpoint ({best_acc:.2f}%) to {args.checkpoint}",
                      flush=True)
            if is_dist:
                dist.barrier()

    if args.save_last and is_main:
        obj = model.module if hasattr(model, "module") else model
        last_path = args.checkpoint.replace(".pt", "_last.pt")
        torch.save(
            {"epoch": args.epochs, "model_state_dict": obj.state_dict(), "args": vars(args)},
            last_path,
        )
        print(f"  -> saved last checkpoint to {last_path}", flush=True)

    if is_main:
        print(f"done. best val acc: {best_acc:.2f}%")
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
