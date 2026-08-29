"""
model.py — 3-layer CNN for MNIST digit classification.

Every layer and tensor operation is written explicitly with ``einops``
(``rearrange`` / ``reduce``) and ``einsum``:

  * Conv2d        -> zero-pad (explicit buffer) + sum of K^2 shifted
                     einsum contractions (one rank-1 conv per kernel tap)
  * MaxPool2d     -> reshape into windows (einops.rearrange) + reduce('max')
  * GlobalAvgPool -> einops reduce('mean') over H and W
  * Linear        -> single einsum contraction
  * Flatten       -> einops.rearrange
  * Zero-padding  -> explicit zero buffer (no F.pad)

No training loop: defines the architecture, verifies shapes, and
cross-checks the einsum conv/linear against PyTorch reference ops.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce


class EinsumConv2d(nn.Module):
    """Conv2d as a sum of K^2 shifted einsum contractions.

    Each kernel tap (u, v) is a rank-1 convolution:
        y_u,v[b, o, h, w] = sum_c weight[o, c, u, v] * x[b, c, h+u, w+v]
    The K^2 partial maps are summed into the final feature map.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "odd kernel sizes only"
        self.kernel = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        fan_in = in_channels * kernel_size * kernel_size
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)  # ReLU-friendly
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def _zero_pad(x: torch.Tensor, p: int) -> torch.Tensor:
        """Zero-pad the spatial dims with an explicit buffer (no F.pad)."""
        if p == 0:
            return x
        b, c, h, w = x.shape
        padded = x.new_zeros(b, c, h + 2 * p, w + 2 * p)
        padded[:, :, p : p + h, p : p + w] = x
        return padded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, H, W)
        if self.padding:
            x = self._zero_pad(x, self.padding)
        if self.stride != 1:
            x = x[..., :: self.stride, :: self.stride]

        k = self.kernel
        h_out = x.shape[-2] - k + 1
        w_out = x.shape[-1] - k + 1

        # Sum of K^2 rank-1 convolutions (one per kernel tap).
        out = None
        for u in range(k):
            for v in range(k):
                w_uv = self.weight[:, :, u, v]  # (O, C)
                x_uv = x[..., u : u + h_out, v : v + w_out]
                y_uv = einsum(x_uv, w_uv, "b c h w, o c -> b o h w")
                out = y_uv if out is None else out + y_uv

        # bias lives on the channel axis: (O,) -> (1, O, 1, 1)
        return out + rearrange(self.bias, "o -> 1 o 1 1")


class EinsumMaxPool2d(nn.Module):
    """MaxPool2d: split spatial dims into windows (einops) + reduce('max')."""

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.kernel = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, C, H_out, W_out, K, K) -> max over the window
        if self.stride != self.kernel:
            raise NotImplementedError("use stride == kernel (non-overlapping)")
        k = self.kernel
        windows = rearrange(x, "b c (h k) (w l) -> b c h w k l", k=k, l=k)
        return reduce(windows, "b c h w k l -> b c h w", "max")


class EinsumGlobalAvgPool2d(nn.Module):
    """Global average pooling: einops reduce('mean') over H and W."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return reduce(x, "b c h w -> b c 1 1", "mean")


class EinsumLinear(nn.Module):
    """Fully connected layer as a single einsum contraction."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)  # ReLU-friendly
        bound = 1 / (in_features**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = rearrange(x, "b ... -> b (...)")
        return einsum(flat, self.weight, "b i, o i -> b o") + self.bias


class MNISTCNN(nn.Module):
    """3-layer CNN for 28x28 grayscale MNIST digits.

    conv1(1->32) -> maxpool -> conv2(32->64) -> maxpool -> conv3(64->128)
    -> global avg pool -> fc(128 -> 10)

    Spatial flow: 28x28 -> 14x14 -> 7x7 -> 1x1
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = EinsumConv2d(1, 32)
        self.pool1 = EinsumMaxPool2d(2, 2)
        self.conv2 = EinsumConv2d(32, 64)
        self.pool2 = EinsumMaxPool2d(2, 2)
        self.conv3 = EinsumConv2d(64, 128)
        self.gap = EinsumGlobalAvgPool2d()
        self.fc = EinsumLinear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.gap(x)
        x = rearrange(x, "b c 1 1 -> b c")
        return self.fc(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MNISTCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MNISTCNN ({device}): {n_params:,} parameters")

    # --- shape check ------------------------------------------------------
    x = torch.randn(4, 1, 28, 28, device=device)
    logits = model(x)
    print(f"input {tuple(x.shape)}  ->  logits {tuple(logits.shape)}")
    assert logits.shape == (4, 10)

    # --- cross-check einsum conv against F.conv2d -------------------------
    import torch.nn.functional as F

    ref = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True).to(device)
    with torch.no_grad():
        ref.weight.copy_(model.conv1.weight)
        ref.bias.copy_(model.conv1.bias)
    diff = (model.conv1(x) - ref(x)).abs().max().item()
    print(f"conv1 vs F.conv2d:  max abs diff = {diff:.3e}")
    assert diff < 1e-5

    # --- cross-check einsum linear against F.linear -----------------------
    ref_fc = nn.Linear(128, 10).to(device)
    with torch.no_grad():
        ref_fc.weight.copy_(model.fc.weight)
        ref_fc.bias.copy_(model.fc.bias)
    feat = torch.randn(4, 128, device=device)
    diff = (model.fc(feat) - ref_fc(feat)).abs().max().item()
    print(f"fc    vs F.linear:  max abs diff = {diff:.3e}")
    assert diff < 1e-5

    print("OK: shapes and numerics verified (no training performed)")
