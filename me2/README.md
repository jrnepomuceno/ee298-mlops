# VCM Runtime

This repository develops and validates an offline voice-command runtime for Raspberry Pi 5, with an M4 Mac as a development and benchmarking machine. The runtime uses a versioned ONNX model, NumPy/ONNX Runtime inference, deterministic task handling, local audio I/O, and optional Piper speech.

## Repository Scope

- **Model loading and ONNX porting:** artifact versions, input/output contracts, compatibility, and runtime loading.
- **Inference:** WAV-to-features-to-intent/slots using the Pi-compatible NumPy implementation.
- **Harness and tasks:** wake/command state, confidence and OOV handling, allowlisted actions, timers, reminders, media, calls, volume, HVAC, and information queries.
- **Microphone and speaker:** capture, VAD, playback, acknowledgement, and device selection.
- **Piper TTS:** offline voice model loading and reply playback.
- **Benchmarking and acceptance:** latency, RTF, quality, memory, and thermal checks on the actual Pi.

Routine work here is runtime integration and validation, not dataset construction or model training. Training-related modules and notes are retained where needed for provenance, reproducibility, or future model handoff; they are not the deployment entry point.

## Runtime Map

- [ONNX inference](inference/README.md): direct model inference and CPU-only dependencies.
- [Pi 5 harness](rpi5/README.md): microphone/replay workflow, task execution, replies, Piper, and optional RGB feedback.
- [ONNX inference guide](ONNX_INFERENCE_GUIDE.md): model input/output contract and loading details.
- [User acceptance benchmark](docs/USER_ACCEPTANCE_BENCHMARK.md): Pi preflight, latency/thermal run, and labeled WAV evaluation.
- [Validation history](docs/VALIDATION.md): recorded results and their model provenance.

The versioned ONNX artifact and its manifest live under `models/onnx/`. Always pass an explicit model path for a recorded benchmark so the artifact under test is unambiguous.