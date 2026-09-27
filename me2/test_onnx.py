#!/usr/bin/env python3
"""Test ONNX export and inference."""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

# Add the project root to the path so we can import our modules
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config
from model import VCM
from utils.model_utils import get_device, load_checkpoint


def test_onnx_export_and_inference(checkpoint_path: str, onnx_path: str) -> int:
    """Test exporting a model to ONNX and running inference with it."""
    # Load the checkpoint
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1
    
    # Use CPU for ONNX export
    device = get_device("cpu")
    
    print(f"[test] Loading model from {ckpt_path}")
    model, payload = load_checkpoint(ckpt_path, device)
    model.eval()
    
    # Create a dummy input tensor with the expected shape
    dummy_input = torch.randn(1, 100, config.N_MELS, device=device)
    
    # Export the model
    print(f"[test] Exporting model to {onnx_path}")
    
    # Try different export approaches
    try:
        torch.onnx.export(
            model,                          # model being run
            dummy_input,                    # model input (or a tuple for multiple inputs)
            onnx_path,                      # where to save the model
            export_params=True,             # store the trained parameter weights inside the model file
            opset_version=11,               # the ONNX version to export the model to
            do_constant_folding=True,       # whether to execute constant folding for optimization
            input_names=['mels'],           # the model's input names
            output_names=['intent_logits', 'ctc_logits'],  # the model's output names
            dynamic_axes={
                'mels': {0: 'batch_size', 1: 'time'},    # variable length axes
                'intent_logits': {0: 'batch_size'},
                'ctc_logits': {0: 'batch_size', 1: 'time'}
            },
            # Add this parameter to handle the GRU issue
            training=torch.onnx.TrainingMode.EVAL,
        )
    except Exception as e:
        print(f"[test] ONNX export failed with default settings: {e}")
        print("[test] Trying alternative export method...")
        
        # Alternative export method with different parameters
        try:
            torch.onnx.export(
                model,                          # model being run
                dummy_input,                    # model input (or a tuple for multiple inputs)
                onnx_path,                      # where to save the model
                export_params=True,             # store the trained parameter weights inside the model file
                opset_version=13,               # newer ONNX version
                do_constant_folding=True,       # whether to execute constant folding for optimization
                input_names=['mels'],           # the model's input names
                output_names=['intent_logits', 'ctc_logits'],  # the model's output names
                dynamic_axes={
                    'mels': {0: 'batch_size', 1: 'time'},    # variable length axes
                    'intent_logits': {0: 'batch_size'},
                    'ctc_logits': {0: 'batch_size', 1: 'time'}
                },
                # Use different settings for better compatibility
                training=torch.onnx.TrainingMode.EVAL,
                # Disable dynamic shapes which might cause issues with GRU
                dynamic_shapes=False,
            )
        except Exception as e2:
            print(f"[test] Alternative ONNX export also failed: {e2}")
            print("[test] Trying with simplified dynamic axes...")
            
            # Simplified export with minimal dynamic axes
            try:
                torch.onnx.export(
                    model,                          # model being run
                    dummy_input,                    # model input (or a tuple for multiple inputs)
                    onnx_path,                      # where to save the model
                    export_params=True,             # store the trained parameter weights inside the model file
                    opset_version=13,               # newer ONNX version
                    do_constant_folding=True,       # whether to execute constant folding for optimization
                    input_names=['mels'],           # the model's input names
                    output_names=['intent_logits', 'ctc_logits'],  # the model's output names
                    dynamic_axes={
                        'mels': {0: 'batch_size'},    # only batch size as dynamic
                        'intent_logits': {0: 'batch_size'},
                        'ctc_logits': {0: 'batch_size'}
                    },
                    training=torch.onnx.TrainingMode.EVAL,
                )
            except Exception as e3:
                print(f"[test] All ONNX export attempts failed: {e3}")
                return 1
    
    print(f"[test] Model exported to {onnx_path}")
    
    # Verify the exported model
    try:
        import onnx
        print("[test] Verifying exported model...")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("[test] Model verified successfully")
    except Exception as e:
        print(f"[test] Model verification failed: {e}")
        return 1
    
    # Test inference with ONNX Runtime
    print("[test] Testing ONNX Runtime inference...")
    try:
        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(onnx_path)
        
        # Create test input
        test_input = np.random.randn(1, 100, config.N_MELS).astype(np.float32)
        
        # Run inference
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        intent_logits, ctc_logits = ort_session.run(None, ort_inputs)
        
        print(f"[test] ONNX Runtime inference successful")
        print(f"[test] Intent logits shape: {intent_logits.shape}")
        print(f"[test] CTC logits shape: {ctc_logits.shape}")
        
        # Compare with PyTorch inference
        print("[test] Comparing with PyTorch inference...")
        with torch.no_grad():
            torch_intent_logits, torch_ctc_logits = model(torch.tensor(test_input))
        
        # Check if results are close
        intent_diff = np.abs(intent_logits - torch_intent_logits.numpy()).max()
        ctc_diff = np.abs(ctc_logits - torch_ctc_logits.numpy()).max()
        
        print(f"[test] Max difference in intent logits: {intent_diff}")
        print(f"[test] Max difference in CTC logits: {ctc_diff}")
        
        if intent_diff < 1e-4 and ctc_diff < 1e-4:
            print("[test] ✅ ONNX and PyTorch results match!")
            return 0
        else:
            print("[test] ❌ ONNX and PyTorch results differ!")
            return 1
            
    except Exception as e:
        print(f"[test] ONNX Runtime inference failed: {e}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Test ONNX export and inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pi5-vcm-best.pt",
                       help="path to trained checkpoint (default: checkpoints/pi5-vcm-best.pt)")
    parser.add_argument("--output", type=str, default="vcm_model.onnx",
                       help="output path for ONNX model (default: vcm_model.onnx)")
    
    args = parser.parse_args()
    
    return test_onnx_export_and_inference(args.checkpoint, args.output)


if __name__ == "__main__":
    raise SystemExit(main())