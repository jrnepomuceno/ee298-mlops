# Pi5-VCM — on-device inference

Self-contained inference folder for the **Raspberry Pi 5**. The input is a
trained checkpoint, **`best.pt`**. Everything runs CPU-only, standalone,
no cloud, no LLM.

## Layout

| File | Purpose |
|---|---|
| `infer.py` | Entry point. Loads `best.pt`, runs a wav (or self-test), prints intent / transcript / slots / latency. |
| `requirements.txt` | Runtime deps: `numpy`, `torch`, `torchaudio`. |
| `run_pi5.sh` | Pi wrapper (uses the project venv if present, else `python3`). |
| `sync.sh` | Re-vendor `config.py` / `model/` / `utils/` from the project root so this folder can ship standalone. |

By default `infer.py` imports the **canonical** `config.py`, `model/`, and
`utils/` from the project root (one level up) — there is a single source of
truth, no vendored copies in this folder. Run `./sync.sh` only if you need to
copy this folder onto the Pi by itself (it vendors `config.py`, `model/`, and
`utils/` next to `infer.py`).

## Quick start (on the Pi)

```bash
./run_pi5.sh --checkpoint best.pt --input cmd.wav
./run_pi5.sh --checkpoint best.pt --self-test --json
```
