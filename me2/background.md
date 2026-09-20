

Planning only — here's the full plan, weighted toward your individual tasks (2, 4, 5, 6) with your defined contributions to the collective ones (1, 3).

## Assumptions (flag these to the team)
1. **"Pure VCM doing 1 to 10"** — I'm reading this as: *your trained model* must do the recognition end-to-end (audio → intent + slots → action), with no LLM and no off-the-shelf ASR doing the heavy lifting. This rules out the Vosk+parser design I suggested earlier — that would make your "trained model" a thin wrapper. If "1 to 10" means something else (e.g., a fixed 10-command set), tell me and I'll adjust.
2. **Command set**: your ~15 commands, including parameterized ones (dim to X%, timer X min, alarm X am/pm, temp X°) and free-form ones (remind me to…, call…). Free-form slots are what force a real (non-LLM) decoder inside the VCM.
3. **Device**: RPi4 or RPi5, 2 GB+ RAM, headless, CPU-only.

## Target architecture (one trained model, ~3–6 MB int8)

```
mic → VAD → openWakeWord (pretrained, ~200KB) → VCM (your model) → parser/actions
                                            ├─ encoder: CNN/GRU over log-mel, ~2–4M params
                                            ├─ head 1: intent classifier (15–20 classes)
                                            └─ head 2: CTC decoder, constrained vocab
                                               (digits 0–120, time words, contact names,
                                                reminder keywords) → slot text
```

- **Why one model with two heads**: the intent head nails the fixed commands; the CTC head (constrained to a small vocabulary — not open language) transcribes the slot content for "dim to **forty** percent", "call **mom**". CTC with a ~200-token vocab is a classic, small, LLM-free transducer — this is the piece that satisfies "pure VCM" while still handling free-form slots.
- **openWakeWord** is pretrained and not an LLM; acceptable as the front door. If the team wants 100% self-built, training a custom wake word is a stretch goal, not a requirement.
- **Everything else is deterministic code**: `word2num`, `parse_time`, contact fuzzy-match, action dispatch. No cloud, no LLM, no external ASR.

## Task 2 — Build & train (individual)

**Model spec**
- Encoder: 4–6 conv blocks (or Conv+GRU) on 80-dim log-mel, 16 kHz, 25 ms frames → ~2–4M params
- Heads: softmax intent (20 classes incl. `unknown`/OOV) + CTC over constrained vocab
- Joint loss: cross-entropy + CTC loss; train in PyTorch (CPU-friendly, or one GPU session if available)

**Training procedure**
1. SpecAugment (freq/time masking) + speed/pitch/noise augmentation (audiomentations)
2. 30–60 epochs, AdamW, LR schedule; early-stop on dev set
3. Export → ONNX → **int8 PTQ** (TFLite or ONNX Runtime) — target ≤6 MB, RTF < 0.3 on RPi4
4. Keep a float32 reference checkpoint for the benchmark

**Milestones**: M1 model trains on CPU in <4 h · M2 dev-set intent acc ≥90% · M3 int8 model on RPi, RTF measured.

## Task 1 — Dataset (collective; your contribution)

Propose the team split so nothing falls through:
- **Recording**: each member records all 15 intents, 100–200 utterances each, varied phrasing ("dim the lights to 40 percent" / "set lights at 40%"); minimum **3–5 distinct speakers per intent** (speaker-disjoint test set is what makes the benchmark credible)
- **Slot coverage**: explicit digit/time/name matrix — every intent with a slot × a spread of values (0–100%, 1–120 min, all clock patterns incl. "half past", "quarter to", am/pm)
- **Negatives**: silence, ambient noise, out-of-vocabulary speech (chatter, other commands) — needed for the OOV class and FAR measurement
- **Your slice**: define the annotation schema (`utterance_id, speaker, room, snr, intent, slots_json, transcript`), the recording script/CLI that captures + auto-labels, and the augmentation pipeline. This makes you the dataset's structural owner, which pays off in task 3.

## Task 3 — Benchmark (collective; your contribution)

Propose this rubric to the group:
| Dimension | Metric |
|---|---|
| Intent accuracy | Overall acc, per-intent F1, macro-F1 |
| Slot accuracy | Exact-match on numbers/times; digit-error rate; time-parse success |
| Robustness | Clean vs SNR 0/5/10 dB; per-speaker (held-out speakers) |
| Rejection | OOV rejection rate, false-accept rate |
| Runtime | RTF, p50/p95 end-to-end latency (wake→action), peak RAM, model size, power draw |
| Wake word | FAR (false accepts/hr), FRR |

**Protocol**: fixed held-out set (speaker-disjoint + noisy), identical hardware (agreed RPi model), int8 build, 5-run average. **Your contribution**: the eval harness — one script that loads any team member's model + dataset split and emits the full metric table, so everyone's results are comparable. That's the highest-leverage thing you can build for the group.

## Task 4 — Validate (individual)

1. Run your model through the shared harness on the held-out set; report the full table
2. Ablations: with/without augmentation, with/without slot head, float32 vs int8 (quantization cost)
3. Error analysis: confusion matrix (which intents collide — expect *turn on/off lights*, *pause/stop*), worst slot errors, noisy-condition degradation curve
4. Latency profile on the actual demo device (perf counters, not just CPU bench)
5. Write-up: results vs. team baseline, limitations, what you'd change

## Task 5 — Real-world demo (individual)

- **Hardware**: RPi4/5 (shared pool — book yours early), USB or ReSpeaker mic, speaker for TTS reply (Piper, offline), optional LED strip/GPIO to show light actions
- **Pipeline**: wake word → VCM → parser → action dispatch:
  - music: MPD/`mpc` · lights/temp: MQTT → Home Assistant (or GPIO sim if no HA) · timer/alarm: scheduler + chime · reminders: `reminders.json` · call: dialer stub (simulated dial is fine for demo) · time/weather: local clock + cached/offline source (rule 7 — no cloud; use a local mock or pre-fetched data, and say so)
- **Demo script** (2 min): wake → "dim lights to 40%" (LED actually dims) → "set timer for 2 minutes" (countdown + chime) → "remind me to water the plants" → "what are my reminders" (TTS reads back) → noisy-room segment to show robustness
- **On-screen**: live transcript + intent + slots + latency per command — this makes the "real-time, tiny, standalone" claims visible to graders

## Task 6 — Tiny + real-time (individual)

Budget to commit to publicly:
- Model ≤ **6 MB int8**; total app RAM ≤ **1.2 GB** on 2 GB board (measured, not estimated)
- **RTF ≤ 0.3** on RPi4 (worst-case device in the pool) → end-to-end wake→intent p95 < 500 ms
- Proof: `time`/perf logs from the demo device included in the write-up; int8 vs float32 comparison

## Tasks 7 & 8 — Constraints (how they're satisfied)
- **Standalone/no cloud**: all components local (openWakeWord ONNX, your ONNX model, Piper TTS, local storage); network disabled during demo as proof
- **No LLM**: the VCM is a CNN/GRU + classifier + constrained-CTC; everything downstream is deterministic code. State this explicitly in the write-up with the param count.

## Timeline (adjust to your deadline)
| Week | Deliverable |
|---|---|
| 1 | Dataset schema + recording CLI (task 1); team starts recording; model skeleton trains on toy data |
| 2 | Full dataset assembled + augmented; VCM v1 trained; benchmark rubric agreed (task 3) |
| 3 | Eval harness shipped to team; VCM v2 (augmented, tuned); int8 export + RPi profiling |
| 4 | Demo wired end-to-end; validation run + ablations + error analysis (task 4); write-up draft |
| 5 | Demo rehearsal, latency polish, final write-up |

## Top risks
1. **Free-form slots are the hard 10%** — mitigate by constraining the CTC vocab tightly and over-collecting slot-value variety in week 1
2. **Dataset quality** (team-dependent) — mitigate by owning the recording tooling so bad data is rejected at capture
3. **Shared-device contention** for demos — book RPi4 (the weaker box) for your validation so your real-time claim is defensible
4. **Weather/call need external services** — pre-agree with the team that demo uses local mocks; rule 7 forces this anyway

Two things to confirm with the team before week 1: the exact command list (15 vs a trimmed 10), and whether pretrained openWakeWord counts as allowed or the wake word must also be self-trained. Want me to turn this into a written plan doc (markdown) you can share with the group?

workspace/
├── pyproject.toml
├── requirements.txt
├── main.py
├── data/
│   ├── audio/
│   └── labels/
├── models/
│   └── vcm_model.onnx
└── utils/
    ├── audio_utils.py
    └── model_utils.py