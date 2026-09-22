# ME2 — Voice Controlled Smart Device: Project Plan

**Last updated:** 2026-09-19
**Target device:** Raspberry Pi 5 (2 GB profile; actual unit 8 GB)
**Hard constraints:** ≤ 6 MB int8 model · RTF ≤ 0.3 (near-instant) · fully on-device · no cloud · no LLM

---

## 1. Problem statement

ASR models are too large for on-device use. We build a **tiny Voice Command Model (VCM)** that
maps audio directly to the ~10 most common smart-device commands, with slot extraction, in one
trained model:

```
log-mel (T, 80) → 2D conv stack (T ÷ 4) → Linear(10240→128) → BiGRU (2×128)
    → intent head : 16-way classification (15 commands + OOV)
    → slot head   : CTC over 77-token constrained vocab → rule-based parse_slots()
```

The constrained CTC vocabulary (digits + clock/slot words) is what replaces an LLM: decoding
yields a transcript, and deterministic regexes turn (intent, transcript) into slots.

### Intent taxonomy (16 classes)

`turn_on_lights`, `turn_off_lights`, `dim_lights`, `set_temperature`, `play_music`,
`pause_music`, `stop_music`, `set_timer`, `set_alarm`, `cancel_timer`, `remind`, `call`,
`what_time`, `what_weather`, `what_reminders`, `oov`

---

## 2. Current status (verified by execution)

### ✅ Done

| Item | Evidence |
|---|---|
| Full pipeline runs end-to-end | `generate` → `train` → `test` → `demo` all exit 0 |
| Fatal bug 1 fixed — conv/GRU shape mismatch | `MaxPool2d((2,1))` ×2 in `model.py`; forward passes: `intent (2,16)`, `ctc (2,100,77)` |
| Fatal bug 2 fixed — `KeyError: 'acc'` | `val_m["accuracy"]` in `main.py` (history + print) |
| Model size within budget | **1,958,109 params ≈ 1.96 MB int8** (budget: 6 MB) via `gru_in = Linear(10240, 128)` |
| Inference speed (Mac/MPS) | ~31 ms/utterance steady-state on 400-frame input |
| Minor cleanups | dead `:00` conditional, wrapper deletion, help text, `MAX_DURATION_S` removal, cache annotation |
| Code synced to `~/MyPlayground/Pi5-VCM/` | 5 files diff-verified byte-identical |

### ⚠️ Known issues (from second analysis pass)

1. **`parse_slots` multi-digit fragility** — CTC tokenizes digits per character ("40" → "4", "0"),
   so the decoded transcript has *no spaces between digits*; regexes like `(\d+)\s*percent` still
   match, but `call\s+(\w+)` and `remind me (?:to|about) (.+)` will mis-extract on multi-word
   content. Needs a digit-merge step in `parse_slots` or the decoder.
2. **OOV samples poison the CTC head** — OOV transcripts tokenize to mostly `<unk>`; CTC loss
   trains on that noise. Fix: mask the CTC term when `intent == "oov"`.
3. **Intent pooling ignores padding** — `_pool()` means over zero-padded frames. Fix:
   length-aware masked mean (collate already has lengths available).
4. **CTC `input_lengths` uses padded length**, not true utterance length.
5. **Demo pads every utterance to 400 frames** — the single biggest latency lever for
   "instantaneous" on the Pi. Fix: infer on true length (variable-length batch or single
   utterance), or VAD-trim before inference.
6. **Noisy-condition robustness untested** (brief task 3) — no noise augmentation in the
   synthetic generator yet.
7. **No `requirements.txt`**, no mic/VAD/wake-word/action-dispatch layer (brief task 5).

### 📌 Environment notes

- Training venv: `~/MyPlayground/Pi5-VCM/pi5venv/bin/python` (torch 2.14, torchaudio 2.11)
- Old checkpoints are **invalid** against the new architecture (`gru_in` layer) — retrain fresh.
- UPD HPC (`hpc.coe.upd.edu.ph` → 10.153.60.100) reachable via VPN, port 22 open, but
  **key auth still denied** — public key `SHA256:VjlyfnD45JafiXns+FoOKLE9rbvkptDXpck2gtlkoNo`
  apparently not in `authorized_keys` (or registered under a different user/host). Optional
  accelerator for long training runs; **not required** — the model is small enough that the
  Mac's MPS handles it (~12–25 s/epoch on synthetic data).

---

## 3. Plan by brief task

### Task 1 — Dataset (collective) — 🟡 in progress

Current state: synthetic formant-tone generator (`main.py generate`) for pipeline dev only.

**Next steps:**
1. **Record real speech** — the group's collective dataset. Per intent, target:
   - ≥ 30 utterances × ≥ 5 speakers × 2–3 phrasings each (≈ 300–450 clips/intent)
   - Record 16 kHz mono WAV; quiet room; 0.3–0.5 s leading/trailing silence
   - Include OOV negatives: "tell me a joke", "who won the game", "set the oven", etc.
   - Include slot variations: "dim lights to 30 percent", "set a timer for 5 minutes",
     "set an alarm for 7 am", "set temperature to 24 degrees"
2. **Build the manifest** — CSV/JSONL with `path, intent, transcript, speaker` (the
   `dataset.py` real-data path already supports this; no code change needed).
3. **Augment** — add to the generator: background noise (babble/music at SNR 5–15 dB),
   speed perturbation (0.9–1.1×), random gain. This also feeds the noisy benchmark (task 3).
4. **Split** — speaker-disjoint train/val/test (no speaker in both train and test).

### Task 2 — Build & train the VCM (individual) — 🟡 model built, needs real training

Current state: model architecture finalized (1.96 M params). Trained only 2 epochs on toy
synthetic data (acc 0.155 — pipeline check only).

**Next steps:**
1. Apply the 4 training fixes (known issues #1–#4 above):
   - digit-merge in `parse_slots`/decoder
   - CTC loss mask for OOV samples (`train.py`)
   - length-aware masked mean in `_pool` (`model.py`)
   - true-length `input_lengths` for CTC (`train.py`/`dataloader.py`)
2. **Train on the real dataset** — Adam, lr 1e-3 with cosine decay, 30–50 epochs,
   early-stop on val intent accuracy. ~15 s/epoch on Mac MPS; HPC optional.
3. **Export for the Pi** — TorchScript (`torch.jit.script`) or ONNX; verify int8 size
   ≤ 6 MB and parity with PyTorch on 20 held-out clips.
4. **Latency optimization** — remove pad-to-400 (issue #5); measure per-utterance latency
   on true lengths. Target: < 100 ms on Pi 5 for a typical 1–2 s command.

### Task 3 — Benchmark design (collective) — 🔴 not started

Proposed benchmark (draft for the group):
- **Clean set** — held-out real recordings, speaker-disjoint.
- **Noisy set** — same clips + noise at SNR 15 / 10 / 5 dB.
- **Metrics:**
  - Intent accuracy (overall + per-intent confusion matrix)
  - Slot F1 (exact slot match per command)
  - OOV rejection rate (OOV clips correctly flagged, in-vocab not rejected)
  - **RTF** = inference time / audio duration (must be ≤ 0.3 on Pi 5)
  - Latency p50/p95 per utterance
- **Hardware matrix** — report on Pi 4 (4 GB) and Pi 5 (2 GB / 8 GB) to cover the brief's
  "RPi4 or 5" requirement.

### Task 4 — Validate performance (individual) — 🔴 not started

1. Run the benchmark (task 3) on the final model; record all metrics.
2. Ablation: clean vs. noisy; with/without augmentation.
3. Document: `VALIDATION.md` with confusion matrix, per-intent accuracy, RTF, latency
   percentiles, and the int8 size measurement.

### Task 5 — Real-world demo (individual) — 🟡 harness started

Implemented the first Pi-local harness slice in `rpi5/`:
- replay/self-test WAV input through the existing `inference/infer.py` path
- optional default-microphone capture with a small energy-based VAD
- structured JSON command events with request ID, timestamp, result, and action
- confidence/OOV gate with an allowlisted dry-run dispatcher
- Pi-side MCP adapter with `health`, `recognize_file`, and `listen_once` tools
- optional DuoCast RGB feedback: black idle, fast cyan wake wave, cyan processing
   wave, green TTS wave, 500 ms hold, black idle return
- optional Piper TTS playback through the Pi's configured wireless speaker
- combined persistent wake-word + VCM harness with static WAV replies
- no microphone, network, GPIO, MQTT, or real side effects yet

Target runtime layout:

```
microphone -> VAD -> VCM harness -> MCP server -> deterministic reply + action
                                                   -> optional offline TTS
```

The MCP server will run on the Pi alongside the harness. It will expose local
tools such as `listen_once`, `recognize_audio`, `execute_command`, and `health`.
Alexa-like replies will initially be generated from deterministic templates
(`"The lights are on"`, `"Timer set for 5 minutes"`, etc.) and optionally spoken
with an offline TTS engine such as Piper. No internet or LLM is required.

Next steps:

Next runtime work:
1. **Audio capture** — test `sounddevice` at 16 kHz with the actual USB mic.
2. **VAD** — tune energy thresholds against room noise; optional `webrtcvad`;
   optional openWakeWord for "Hey VCM" trigger (optional stretch).
3. **Inference** — load exported model, run on VAD-trimmed audio (no pad-to-400).
4. **Action dispatch** — intent+slots → mock device actions (print/log, or MQTT to a
   light switch / `spidev` GPIO on the Pi).
5. **Feedback** — beep + LED on recognized/rejected.

### Task 6 — Tiny & real-time (individual) — 🟡 on track

- ✅ Size: 1.96 MB int8 (budget 6 MB)
- ⏳ RTF: must be measured **on the Pi** (Mac number: ~31 ms/400 frames; after removing
  padding, expect well under 100 ms for 1–2 s commands on Pi 5)
- ⏳ 2 GB profile check: model + runtime memory footprint on the 2 GB config

### Task 7 — Standalone (individual) — ✅ by design

No network calls anywhere in the pipeline. Verify at demo time: run the Pi with Wi-Fi
disabled and confirm full operation.

### Task 8 — No LLM (individual) — ✅ by design

Single trained model + rule-based `parse_slots`. Nothing else.

---

## 4. Immediate next steps (this week)

1. **[me]** Apply the 4 training fixes (known issues #1–#4) to `model.py`, `train.py`,
   `dataloader.py` — small, surgical changes.
2. **[me]** Add microphone/VAD adapters on top of `rpi5/`, keeping replay as the smoke-test
   path and dry-run actions as the default.
3. **[me]** Remove pad-to-400 from `demo`/`test` inference paths (issue #5); re-measure
   latency on true lengths.
4. **[me]** Write `requirements.txt` (torch, torchaudio, numpy, sounddevice, webrtcvad).
5. **[you + group]** Start real recording (task 1) — this is the long pole; the synthetic
   generator is only a pipeline stand-in.
6. **[me]** Once ≥ first batch of real recordings exists: build manifest, train 30–50
   epochs, export TorchScript/ONNX.
7. **[me]** Add the Pi-side MCP adapter over the harness event contract; use stdio
   for a local MCP client or a bound local/LAN HTTP transport when needed.
8. **[me]** Draft the benchmark spec (task 3) for the group's collective review.
9. **[optional]** Resolve HPC key auth (register `SHA256:VjlyfnD45JafiXns+FoOKLE9rbvkptDXpck2gtlkoNo`)
   only if Mac training becomes a bottleneck — it probably won't for a 2 M-param model.

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Real speech much harder than synthetic formant tones | Augmentation (noise, speed, gain); expect to iterate on architecture only if val acc stalls |
| Slot accuracy poor on multi-word content (contacts, reminders) | Digit-merge fix + expand CTC vocab contacts; rule parser is trivially extensible |
| Pi 5 2 GB memory pressure with torch runtime | TorchScript/ONNX Runtime; if needed, drop to ONNX int8 quantization (model is 1.96 MB — headroom is huge) |
| Group dataset quality varies | Speaker-disjoint split + per-speaker reporting in the benchmark |
| "Instantaneous" expectation | Remove padding (biggest win), VAD-trim, report p50/p95 honestly in VALIDATION.md |

## 6. Deliverables checklist

- [ ] `plan.md` (this file)
- [ ] Fixed training code (issues #1–#5)
- [ ] `requirements.txt`
- [ ] Real dataset manifest + recordings (group)
- [ ] Trained model + exported artifact (TorchScript/ONNX)
- [ ] Benchmark spec (group) + `VALIDATION.md` with full metrics
- [ ] `rpi5/` harness — replay/self-test → VCM → dry-run action dispatch
- [ ] `demo_pi.py` or microphone adapter — mic → VAD → VCM → action dispatch on Pi 5
- [ ] RTF ≤ 0.3 + latency report on Pi 4 and Pi 5
- [ ] Standalone verification (Wi-Fi off demo)
