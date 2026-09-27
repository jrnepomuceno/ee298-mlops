#!/usr/bin/env python3
"""Export a trained VCM model to ONNX format.

Usage:
    python export_onnx.py --checkpoint checkpoints/pi5-vcm-best.pt --output vcm_model.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Add the project root to the path so we can import our modules
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config
from model.model import VCM
from utils.model_utils import get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export VCM model to ONNX")
    p.add_argument("--checkpoint", type=str, default="checkpoints/pi5-vcm-best.pt",
                   help="path to trained checkpoint (default: checkpoints/pi5-vcm-best.pt)")
    p.add_argument("--output", type=str, default="vcm_model.onnx",
                   help="output path for ONNX model (default: vcm_model.onnx)")
    p.add_argument("--opset", type=int, default=11,
                   help="ONNX opset version (default: 11)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    
    # Load the checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1
    
    # Use CPU for ONNX export
    device = get_device("cpu")
    
    print(f"[onnx] Loading model from {ckpt_path}")
    model, payload = load_checkpoint(ckpt_path, device)
    model.eval()
    
    # Get config from checkpoint or fall back to defaults
    cfg = payload.get("config", {}) or {}
    intents = cfg.get("intents") or config.INTENTS
    ctc_vocab = cfg.get("ctc_vocab") or config.CTC_VOCAB
    
    print(f"[onnx] Model loaded with {len(intents)} intents and {len(ctc_vocab)} CTC tokens")
    
    # Create a dummy input tensor with the expected shape
    # Using a typical mel spectrogram size: (batch_size=1, time_frames=100, n_mels=80)
    dummy_input = torch.randn(1, 100, config.N_MELS, device=device)
    
    # Export the model
    output_path = args.output
    print(f"[onnx] Exporting model to {output_path}")
    
    torch.onnx.export(
        model,                          # model being run
        dummy_input,                    # model input (or a tuple for multiple inputs)
        output_path,                    # where to save the model
        export_params=True,             # store the trained parameter weights inside the model file
        opset_version=args.opset,       # the ONNX version to export the model to
        do_constant_folding=True,       # whether to execute constant folding for optimization
        input_names=['mels'],           # the model's input names
        output_names=['intent_logits', 'ctc_logits'],  # the model's output names
        dynamic_axes={
            'mels': {0: 'batch_size', 1: 'time'},    # variable length axes
            'intent_logits': {0: 'batch_size'},
            'ctc_logits': {0: 'batch_size', 1: 'time'}
        }
    )
    
    print(f"[onnx] Model exported to {output_path}")
    
    # Verify the exported model
    try:
        import onnx
        print("[onnx] Verifying exported model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("[onnx] Model verified successfully")
    except ImportError:
        print("[onnx] ONNX verification skipped (onnx package not installed)")
    except Exception as e:
        print(f"[onnx] Model verification failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())