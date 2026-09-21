# Recording session

Record real 16 kHz mono speech for the VCM training set, straight from a
laptop mic (e.g. a USB HyperX). The tool prompts you with the exact command
phrases (from `dataset.py`), captures each one with a small energy-based VAD,
and writes WAV files plus a manifest that plugs directly into training.

## Setup (once)

```bash
cd ~/MyPlayground/Pi5-VCM/me2
../pi5venv/bin/pip install -r recording/requirements.txt
../pi5venv/bin/python recording/record.py --list-devices   # find your mic
```

Pick the input device index that is your HyperX mic.

## Record

```bash
PY=../pi5venv/bin/python
$PY recording/record.py --list-devices          # (1) find the mic index
$PY recording/record.py --dry-run --speaker joven   # (2) preview phrases
$PY recording/record.py --speaker joven --device 3  # (3) record
```

For each phrase the tool: beeps → you speak → it auto-stops after ~0.5 s of
silence → saves `data/<speaker>/<intent>_NNN.wav` → updates the manifest.
Keys: **Enter** = record, **r** = retry phrase, **s** = skip intent, **q** = quit.

## Coverage targets (from the plan)

- **≥ 30 utterances per intent, from ≥ 5 speakers** (defaults: `--target 30`,
  `--per-speaker 6` → 5 speakers × 6 = 30).
- 16 intents total: 15 commands + `oov` (negatives like "tell me a joke").
- Have each speaker run with their own `--speaker` id. The tool skips intents
  that already meet the target, so sessions resume cleanly.

```bash
$PY recording/record.py --speaker sara --device 3      # next speaker
$PY recording/record.py --status                       # check coverage
```

## Output

```
recording/data/
├── joven/turn_on_lights_001.wav   # 16 kHz mono PCM-16
├── ...
├── manifest.jsonl                 # path, intent, transcript, speaker
└── manifest.csv                   # same, as CSV
```

Paths in the manifest are relative to the project root (`me2/`), so training
from the `me2/` directory works as-is.

## Train

```bash
$PY main.py train --manifest recording/data/manifest.jsonl
```

## Tips

- **Quiet room**, sit ~20–30 cm from the mic, speak at normal pace.
- If VAD cuts you off or starts late, try `--fixed 2.5` (fixed 2.5 s capture,
  no VAD) or tune `--silence` / `--onset`.
- Repeat phrases are allowed (the tool flags them); they just add robustness.
- Slot phrases (timers, alarms, dim %, temperature) are auto-varied with
  different numbers so the CTC slot head sees variety.
