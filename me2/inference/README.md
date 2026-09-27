# Pi5-VCM — on-device inference

Self-contained inference folder for the **Raspberry Pi 5**. The input is a
trained checkpoint, **`best.pt`**. Everything runs CPU-only, standalone,
no cloud, no LLM.

## Layout

| File | Purpose |
|---|---|
| `infer.py` | Entry point. Loads `best.pt`, runs a wav (or self-test), prints intent / transcript / slots / latency. |
| `config.py`, `model.py`, `utils/` | **Vendored copies** of the project-root modules (see `sync.sh`). |
| `requirements.txt` | Runtime deps: `numpy`, `torch`, `torchaudio`. |
| `run_pi5.sh` | Pi wrapper (uses the project venv if present, else `python3`). |
| `sync.sh` | Re-copy `config.py` / `model.py` / `utils/` from the project root. |

## Quick start (on the Pi)


