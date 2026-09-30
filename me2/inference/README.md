# ONNX Inference and Benchmarking

The Raspberry Pi runtime uses the versioned INT8 ONNX model with ONNX Runtime's CPU provider. Audio features are computed by the NumPy Kaldi-fbank implementation, so the deployed inference path does not require PyTorch or torchaudio.

## Layout

| File | Purpose |
|---|---|
| `ort_infer.py` | Pi inference backend. Loads ONNX, extracts features, and returns intent/transcript/slots. |
| `features.py` | Torch-free WAV loading and NumPy Kaldi-fbank implementation shared with the Pi path. |
| `benchmark.py` | Synthetic latency/thermal tests and labeled WAV quality/UAT reports. |
| `infer.py` | PyTorch checkpoint reference path for development; not the lightweight Pi runtime. |
| `run_pi5.sh` | Legacy wrapper for `infer.py`; the `rpi5` harness calls `ort_infer.py` directly. |
| `requirements.txt` | Legacy PyTorch inference dependencies. For Pi ONNX runtime use root `requirements-runtime.txt` or install `numpy` and `onnxruntime`. |
| `sync.sh` | Legacy vendoring helper for standalone copies; the main repository uses canonical root modules. |

`ort_infer.py` imports canonical `config.py`, `model/slots.py`, and `model/onnx_deploy.py` from the project root. Keep those files and the versioned ONNX artifact available together.

## Quick start (on the Pi)

```bash
python -m inference.ort_infer --checkpoint models/onnx/v19-20260929-2/vcm_model_int8.onnx --input cmd.wav
python -m inference.ort_infer --checkpoint models/onnx/v19-20260929-2/vcm_model_int8.onnx --self-test --json
```

For Pi 5 performance and user acceptance procedures, see [`docs/USER_ACCEPTANCE_BENCHMARK.md`](../docs/USER_ACCEPTANCE_BENCHMARK.md).
