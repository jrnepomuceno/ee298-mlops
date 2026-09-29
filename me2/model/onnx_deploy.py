"""ONNX ground-truth packaging for VCM.

Every machine (m4, daniel-pc, the Pi) loads the *same* ONNX artifact. To make
that provable, each export is written into a versioned folder under
``models/onnx/`` and committed to git as the single source of truth::

    models/onnx/
      v19-20260929/
        vcm_model.onnx          # FP32 export (source of the quantized model)
        vcm_model_int8.onnx     # dynamic int8 (what the Pi / CPU paths load)
        manifest.json           # provenance: what produced this, how good it is

``manifest.json`` carries the manifest fingerprint, the source-checkpoint hash,
the intent/CTC schema, the export/quant settings, library versions, and the
final train/val/test metrics -- so any machine can verify it is running the
intended model without guessing.

The resolver (:func:`resolve_latest_onnx`) is deliberately torch-free so the
on-device path (no torch) can import it to find the newest committed build.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import config

# Repo root = parent of model/.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ONNX_DIR = REPO_ROOT / "models" / "onnx"

FP32_NAME = "vcm_model.onnx"
INT8_NAME = "vcm_model_int8.onnx"
MANIFEST_NAME = "manifest.json"


# --------------------------------------------------------------- versioning --
_TAG_RE = re.compile(r"^v(\d+)-(\d{8})(?:-(\d+))?$")


def derive_version_tag(num_intents: int | None = None,
                       date_str: str | None = None) -> str:
    """Default tag: ``v{num_intents}-{YYYYMMDD}`` (e.g. ``v19-20260929``)."""
    ni = num_intents if num_intents is not None else config.NUM_INTENTS
    ds = date_str or datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"v{ni}-{ds}"


def _tag_sort_key(tag: str) -> tuple:
    m = _TAG_RE.match(tag)
    if not m:
        return (0, 0, 0)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))


def next_version_tag(onnx_dir: Path | str,
                     num_intents: int | None = None,
                     date_str: str | None = None) -> str:
    """Pick a unique tag for today: ``v{n}-YYYYMMDD``, then ``-2``, ``-3``, ...

    Keeps the default human-readable while guaranteeing no collision if several
    models are cut on the same day.
    """
    onnx_dir = Path(onnx_dir)
    base = derive_version_tag(num_intents, date_str)
    if not onnx_dir.exists():
        return base
    existing = {p.name for p in onnx_dir.iterdir() if p.is_dir()}
    if base not in existing:
        return base
    seq = 2
    while f"{base}-{seq}" in existing:
        seq += 1
    return f"{base}-{seq}"


# ----------------------------------------------------------------- resolver --
def resolve_latest_onnx(onnx_dir: Path | str = DEFAULT_ONNX_DIR,
                        kind: str = "int8") -> Path | None:
    """Return the newest committed ONNX of ``kind`` ('fp32'|'int8'), or None.

    Scans ``onnx_dir`` for version folders (matching :data:`_TAG_RE`) and picks
    the highest (num_intents, date, seq) that actually contains the artifact.
    Torch-free: safe to import on the Pi.
    """
    onnx_dir = Path(onnx_dir)
    if not onnx_dir.is_dir():
        return None
    fname = INT8_NAME if kind != "fp32" else FP32_NAME
    candidates: list[tuple[tuple, Path]] = []
    for p in onnx_dir.iterdir():
        if not p.is_dir() or not _TAG_RE.match(p.name):
            continue
        artifact = p / fname
        if artifact.exists():
            candidates.append((_tag_sort_key(p.name), artifact))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


def latest_version_tag(onnx_dir: Path | str = DEFAULT_ONNX_DIR) -> str | None:
    """Version tag of the newest folder (by name), or None."""
    onnx_dir = Path(onnx_dir)
    if not onnx_dir.is_dir():
        return None
    dirs = [p.name for p in onnx_dir.iterdir()
            if p.is_dir() and _TAG_RE.match(p.name)]
    if not dirs:
        return None
    return max(dirs, key=_tag_sort_key)


# -------------------------------------------------------------- provenance --
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions() -> dict:
    out = {}
    try:
        import torch
        out["torch"] = torch.__version__
    except Exception:
        out["torch"] = None
    try:
        import onnx
        out["onnx"] = onnx.__version__
    except Exception:
        out["onnx"] = None
    try:
        import onnxruntime
        out["onnxruntime"] = onnxruntime.__version__
    except Exception:
        out["onnxruntime"] = None
    return out


def build_provenance(tag: str,
                     checkpoint: Path,
                     manifest_fingerprint: str | None,
                     metrics: dict | None,
                     opset: int,
                     noise_snr: float | None = None,
                     noise_dir: str | None = None,
                     device: str | None = None,
                     extra_args: dict | None = None) -> dict:
    """Assemble the ``manifest.json`` payload for one ONNX build."""
    ckpt = Path(checkpoint)
    return {
        "tag": tag,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema": {
            "num_intents": config.NUM_INTENTS,
            "intents": list(config.INTENTS),
            "ctc_vocab_size": config.CTC_VOCAB_SIZE,
            "n_mels": config.N_MELS,
            "sample_rate": config.SAMPLE_RATE,
        },
        "source_checkpoint": {
            "path": str(ckpt),
            "sha256": _sha256(ckpt) if ckpt.exists() else None,
        },
        "manifest_fingerprint": manifest_fingerprint,
        "training": {
            "device": device,
            "noise_snr": noise_snr,
            "noise_dir": noise_dir,
            "args": extra_args,
        },
        "export": {
            "opset": opset,
            "quant_scheme": "dynamic",
            "weight_type": "QInt8",
            "activation_type": "fp32",
            "fp32_file": FP32_NAME,
            "int8_file": INT8_NAME,
        },
        "metrics": metrics or {},
        "versions": _versions(),
    }


def write_provenance(version_dir: Path | str, provenance: dict) -> Path:
    version_dir = Path(version_dir)
    version_dir.mkdir(parents=True, exist_ok=True)
    out = version_dir / MANIFEST_NAME
    with open(out, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=False)
        f.write("\n")
    return out


def read_provenance(version_dir: Path | str) -> dict | None:
    p = Path(version_dir) / MANIFEST_NAME
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------- export + quant --
def export_and_quantize(checkpoint: Path,
                        version_dir: Path | str,
                        opset: int = 11) -> tuple[Path, Path]:
    """Export FP32 ONNX then dynamic-int8 quantize into ``version_dir``.

    Returns ``(fp32_path, int8_path)``. Imports torch lazily so the torch-free
    resolver above stays importable on the Pi.
    """
    import torch  # noqa: F401  (ensures torch present before export)
    from utils.model_utils import get_device, load_checkpoint

    version_dir = Path(version_dir)
    version_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = version_dir / FP32_NAME
    int8_path = version_dir / INT8_NAME

    device = get_device("cpu")
    print(f"[onnx] loading checkpoint {checkpoint}")
    model, _payload = load_checkpoint(checkpoint, device)
    model.eval()

    dummy = torch.randn(1, 100, config.N_MELS, device=device)
    print(f"[onnx] exporting FP32 -> {fp32_path} (opset {opset})")
    torch.onnx.export(
        model,
        dummy,
        str(fp32_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["mels"],
        output_names=["intent_logits", "ctc_logits"],
        dynamic_axes={
            "mels": {0: "batch_size", 1: "time"},
            "intent_logits": {0: "batch_size"},
            "ctc_logits": {0: "batch_size", 1: "time"},
        },
        # torch>=2.x defaults to the dynamo (torch.export) ONNX exporter,
        # which mis-decomposes the bidirectional GRU (the 128-vs-384 shape
        # bug). The legacy TorchScript exporter handles it correctly.
        dynamo=False,
    )

    print(f"[onnx] quantizing int8 -> {int8_path}")
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(fp32_path), str(int8_path),
                     weight_type=QuantType.QInt8)

    fp_mb = fp32_path.stat().st_size / 1e6
    i8_mb = int8_path.stat().st_size / 1e6
    print(f"[onnx] fp32={fp_mb:.2f} MB  int8={i8_mb:.2f} MB")
    return fp32_path, int8_path


def package_model(checkpoint: Path,
                  onnx_dir: Path | str = DEFAULT_ONNX_DIR,
                  tag: str | None = None,
                  opset: int = 11,
                  manifest_fingerprint: str | None = None,
                  metrics: dict | None = None,
                  noise_snr: float | None = None,
                  noise_dir: str | None = None,
                  device: str | None = None,
                  extra_args: dict | None = None) -> Path:
    """Full pipeline: derive tag, export+quantize, write provenance.

    Returns the version directory (``models/onnx/<tag>/``).
    """
    onnx_dir = Path(onnx_dir)
    tag = tag or next_version_tag(onnx_dir)
    version_dir = onnx_dir / tag
    print(f"[onnx] packaging model -> {version_dir}")
    export_and_quantize(checkpoint, version_dir, opset=opset)
    prov = build_provenance(
        tag=tag,
        checkpoint=checkpoint,
        manifest_fingerprint=manifest_fingerprint,
        metrics=metrics,
        opset=opset,
        noise_snr=noise_snr,
        noise_dir=noise_dir,
        device=device,
        extra_args=extra_args,
    )
    write_provenance(version_dir, prov)
    print(f"[onnx] wrote {MANIFEST_NAME} (tag={tag})")
    return version_dir


if __name__ == "__main__":  # pragma: no cover - manual helper
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="checkpoints/pi5-vcm-best.pt")
    ap.add_argument("--onnx-dir", default=str(DEFAULT_ONNX_DIR))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--opset", type=int, default=11)
    ap.add_argument("--list", action="store_true",
                    help="list committed versions and exit")
    args = ap.parse_args()
    if args.list:
        d = Path(args.onnx_dir)
        if not d.is_dir():
            print(f"(no versions in {d})")
        else:
            for p in sorted(d.iterdir(), key=lambda x: _tag_sort_key(x.name)):
                if p.is_dir() and _TAG_RE.match(p.name):
                    prov = read_provenance(p)
                    fp = (prov or {}).get("manifest_fingerprint")
                    print(f"{p.name:16} fp={(fp or '?')[:12]} "
                          f"int8={'yes' if (p / INT8_NAME).exists() else 'no'}")
        sys.exit(0)
    package_model(Path(args.checkpoint), onnx_dir=args.onnx_dir, tag=args.tag,
                  opset=args.opset)
