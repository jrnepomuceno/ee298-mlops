#!/usr/bin/env python3
"""Create a test audio file for inference testing."""

import numpy as np
import soundfile as sf
from pathlib import Path
import sys

# Add the project root to the path so we can import our modules
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import config
from utils.audio_utils import synthesize_utterance


def main():
    # Create a test utterance
    words = ["hello", "world"]
    speaker = "test_speaker"
    sample_rate = config.SAMPLE_RATE
    
    print(f"Creating test audio with words: {words}")
    print(f"Sample rate: {sample_rate} Hz")
    
    # Synthesize the utterance
    wav = synthesize_utterance(words, speaker, sample_rate, snr_db=20.0, seed=42)
    
    # Save as WAV file using soundfile
    output_path = "test_audio.wav"
    sf.write(output_path, wav, sample_rate)
    
    print(f"Test audio saved to: {output_path}")
    print(f"Audio length: {len(wav) / sample_rate:.2f} seconds")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())