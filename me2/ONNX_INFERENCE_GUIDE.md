# ONNX Inference Guide

This guide explains how to use the exported ONNX model for inference.

## Model Information

- **Input**: Mel spectrogram (batch_size, time, n_mels)
- **Outputs**: 
  - Intent logits (batch_size, num_intents)
  - CTC logits (batch_size, time, num_classes)
- **Opset Version**: 18

## Using the ONNX Model

### 1. Inference Entry Point

`inference/ort_infer.py` is the ONNX Runtime inference entry point (numpy +
onnxruntime only, no torch). `inference/infer.py` is the equivalent torch
`best.pt` path. Both share the same model contract.

