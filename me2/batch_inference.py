#!/usr/bin/env python3
"""Batch inference script using the exported ONNX model."""

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


def run_single_inference(session: ort.InferenceSession, mel_spectrogram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on a single mel spectrogram."""
    # Ensure the input has the right shape (batch_size, time, n_mels)
    if len(mel_spectrogram.shape) == 2:
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
    
    # Run inference
    ort_inputs = {session.get_inputs()[0].name: mel_spectrogram.astype(np.float32)}
    intent_logits, ctc_logits = session.run(None, ort_inputs)
    
    return intent_logits, ctc_logits


def softmax(x, axis=None):
    """Compute softmax values for x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


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
    
    # Create batch of random mel spectrograms for testing
    batch_size = 4
    test_mels = np.random.randn(batch_size, 100, config.N_MELS).astype(np.float32)
    
    print(f"\nRunning inference on {batch_size} samples (one at a time)")
    print(f"Input shape for each sample: {test_mels[0].shape}")
    
    # Run inference on each sample individually
    all_intent_logits = []
    all_ctc_logits = []
    
    for i in range(batch_size):
        intent_logits, ctc_logits = run_single_inference(session, test_mels[i])
        all_intent_logits.append(intent_logits)
        all_ctc_logits.append(ctc_logits)
    
    # Stack results
    intent_logits = np.vstack(all_intent_logits)
    ctc_logits = np.vstack(all_ctc_logits)
    
    print(f"Intent logits shape: {intent_logits.shape}")
    print(f"CTC logits shape: {ctc_logits.shape}")
    
    # Get predicted intents for each sample in the batch
    intent_probs = softmax(intent_logits, axis=1)
    predicted_intents = np.argmax(intent_probs, axis=1)
    
    for i in range(batch_size):
        print(f"Sample {i}: Predicted intent {predicted_intents[i]} (confidence: {intent_probs[i][predicted_intents[i]]:.4f})")
    
    print("\nBatch inference completed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())