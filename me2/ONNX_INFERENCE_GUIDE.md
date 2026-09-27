# ONNX Inference Guide

This guide explains how to use the exported ONNX model for inference.

## Model Information

- **Input**: Mel spectrogram (batch_size, time, n_mels)
- **Outputs**: 
  - Intent logits (batch_size, num_intents)
  - CTC logits (batch_size, time, num_classes)
- **Opset Version**: 18

## Using the ONNX Model

### 1. Simple Inference Script

The `onnx_inference.py` script demonstrates basic usage of the ONNX model with random inputs:

