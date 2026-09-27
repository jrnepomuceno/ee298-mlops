#!/usr/bin/env python3
"""Audio inference script using the exported ONNX model."""

import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys
import argparse

# Add the project root to the path so we can import our modules
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config
from utils.audio_utils import load_wav_mono, mel_spectrogram, pad_or_trim


def load_onnx_model(onnx_path: str) -> ort.InferenceSession:
    """Load the ONNX model."""
    return ort.InferenceSession(onnx_path)


def preprocess_audio(audio_path: str) -> np.ndarray:
    """Load and preprocess an audio file for inference."""
    # Load audio
    wav = load_wav_mono(audio_path)
    
    # Convert to mel spectrogram
    mel = mel_spectrogram(wav)
    
    # Pad or trim to a fixed length (100 frames as in our test)
    # In a real application, you might want to handle variable lengths differently
    mel = pad_or_trim(mel, 100)
    
    # Convert to numpy array and transpose to (time, n_mels) format
    mel_np = mel.numpy()
    
    return mel_np


def softmax(x):
    """Compute softmax values for x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


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
    parser = argparse.ArgumentParser(description="Run inference on audio files using ONNX model")
    parser.add_argument("--model", type=str, default="test_model.onnx", help="Path to ONNX model")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--intent-classes", type=str, default="assets/intent_classes.txt", 
                       help="Path to intent classes file")
    
    args = parser.parse_args()
    
    # Load the ONNX model
    onnx_path = args.model
    if not Path(onnx_path).exists():
        print(f"Error: ONNX model not found at {onnx_path}")
        return 1
    
    print(f"Loading ONNX model from {onnx_path}")
    session = load_onnx_model(onnx_path)
    
    # Load intent classes
    if Path(args.intent_classes).exists():
        with open(args.intent_classes, 'r') as f:
            intent_classes = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(intent_classes)} intent classes")
    else:
        # Fallback to generic class names
        intent_classes = [f"intent_{i}" for i in range(16)]
        print("Using generic intent class names")
    
    # Check if audio file exists
    audio_path = args.audio
    if not Path(audio_path).exists():
        print(f"Error: Audio file not found at {audio_path}")
        return 1
    
    print(f"Processing audio file: {audio_path}")
    
    # Preprocess audio
    try:
        mel_spectrogram = preprocess_audio(audio_path)
        print(f"Audio preprocessed. Mel spectrogram shape: {mel_spectrogram.shape}")
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return 1
    
    # Run inference
    try:
        print("Running inference...")
        intent_logits, ctc_logits = run_inference(session, mel_spectrogram)
        
        print(f"Intent logits shape: {intent_logits.shape}")
        print(f"CTC logits shape: {ctc_logits.shape}")
        
        # Get predicted intent
        predicted_intent_idx = np.argmax(intent_logits, axis=1)[0]
        # Use our own softmax function
        intent_probs = softmax(intent_logits[0])
        predicted_intent_prob = intent_probs[predicted_intent_idx]
        
        if predicted_intent_idx < len(intent_classes):
            predicted_intent = intent_classes[predicted_intent_idx]
        else:
            predicted_intent = f"unknown_intent_{predicted_intent_idx}"
        
        print(f"Predicted intent: {predicted_intent} (confidence: {predicted_intent_prob:.4f})")
        
        # For CTC, we would typically decode the output using a CTC decoder
        # For simplicity, we'll just show the shape here
        print("CTC output generated (for full decoding, use a CTC decoder)")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return 1
    
    print("\nInference completed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())