#!/usr/bin/env python3
"""Simple inference script using the exported ONNX model."""

import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys

# Add the project root to the path so we can import our modules
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config


def load_onnx_model(onnx_path: str) -> ort.InferenceSession:
    """Load the ONNX model."""
    return ort.InferenceSession(onnx_path)


def run_inference(session: ort.InferenceSession, mel_spectrogram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on a mel spectrogram."""
    # Ensure the input has the right shape (batch_size, time, n_mels)
    if len(mel_spectrogram.shape) == 2:
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
    
    # Run inference
    ort_inputs = {session.get_inputs()[0].name: mel_spectrogram.astype(np.float32)}
    intent_logits, ctc_logits = session.run(None, ort_inputs)
    
    return intent_logits, ctc_logits


def main():
    # Load the ONNX model
    onnx_path = "test_model.onnx"
    if not Path(onnx_path).exists():
        print(f"Error: ONNX model not found at {onnx_path}")
        return 1
    
    print(f"Loading ONNX model from {onnx_path}")
    session = load_onnx_model(onnx_path)
    
    # Print model info
    print("Model inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape}")
    
    print("Model outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape}")
    
    # Create a random mel spectrogram for testing
    # Shape: (time, n_mels) - we'll add batch dimension in run_inference
    test_mel = np.random.randn(100, config.N_MELS).astype(np.float32)
    
    print(f"\nRunning inference on test input with shape {test_mel.shape}")
    intent_logits, ctc_logits = run_inference(session, test_mel)
    
    print(f"Intent logits shape: {intent_logits.shape}")
    print(f"CTC logits shape: {ctc_logits.shape}")
    
    # Get predicted intent (argmax of logits)
    predicted_intent = np.argmax(intent_logits, axis=1)
    print(f"Predicted intent index: {predicted_intent[0]}")
    
    # Get CTC predictions (argmax of logits)
    ctc_predictions = np.argmax(ctc_logits, axis=2)
    print(f"CTC predictions shape: {ctc_predictions.shape}")
    print(f"First 10 CTC predictions: {ctc_predictions[0][:10]}")
    
    print("\nInference completed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())