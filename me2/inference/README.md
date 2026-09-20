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

```bash
cd inference
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/python infer.py --checkpoint best.pt --self-test      # smoke test, no audio needed
./venv/bin/python infer.py --checkpoint best.pt --input cmd.wav  # real 16 kHz mono wav
./run_pi5.sh --checkpoint best.pt --input cmd.wav --json         # machine-readable
```

## Output

For each utterance: predicted **intent** (16-way: 15 commands + `oov`), its
confidence, the **transcript** from the constrained CTC head, the rule-based
**slots** (`parse_slots` — the LLM replacement), the frame count, and the
steady-state **latency** in ms (after `--warmup` passes).

## Notes

- **True-length inference.** Utterances are *trimmed* to `--max-frames`
  (default 400 = 4 s), never zero-padded to 400. A 1.2 s command runs 120
  frames, which is the main on-device latency lever for the "instant"
  requirement.
- **Device.** `--device auto` → `cpu` on the Pi (no CUDA/MPS). Override with
  `--device cpu` if you want to be explicit.
- **Checkpoint contract.** `best.pt` is the payload written by
  `utils/model_utils.save_checkpoint`: `model_state` + a `config` block
  carrying `intents` and `ctc_vocab`, so inference always matches training.
- **Vendored files.** `config.py` / `model.py` / `utils/` are copies. After
  editing those at the project root, run `./sync.sh` to refresh them here.
- **Self-test audio** is synthetic formant tones (a pipeline stand-in), not
  real speech — use it to verify the path, not to judge accuracy.
