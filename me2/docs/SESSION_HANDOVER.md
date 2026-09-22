# Pi5-VCM Session Handover

**Date:** 2026-09-22
**Target:** Raspberry Pi 5 at `192.168.68.52`
**Pi user:** `jdrnepomuceno9`
**Pi runtime:** `~/pi5-vcm`

## Current Architecture

```text
DuoCast microphone
  -> capture queue / VAD
  -> openWakeWord ONNX wake detector
  -> static yes.wav acknowledgement
  -> command capture
  -> persistent PiHarness / VCM checkpoint
  -> intent + constrained slots
  -> action dispatcher
  -> static intent WAV reply
  -> WILLEN speaker
```

The intended state machine is:

```text
STANDBY -> WAKE -> LISTENING -> PROCESSING -> ACTING -> CONFIRMING -> STANDBY
```

The VCM and wake-word model are loaded once by the production `rpi5.run` process. The old `placeholder.py` path has been removed.

## Important Runtime Files

- `rpi5/run.py` - production microphone/VCM harness entry point.
- `rpi5/state_machine.py` - injected, testable lifecycle orchestration.
- `rpi5/audio.py` - capture queue, 44.1 kHz DuoCast input, resampling to 16 kHz, energy VAD, cooldown.
- `rpi5/wakeword.py` - pretrained openWakeWord ONNX adapter with score logging and 1280-sample windows.
- `rpi5/harness.py` - persistent VCM checkpoint loading and inference events.
- `rpi5/replies.py` - intent-to-static-WAV mapping.
- `rpi5/tts.py` - static WAV playback through `pw-play`/`afplay`.
- `rpi5/rgb.py` - DuoCast RGB lifecycle control.
- `tests/test_state_machine.py` - state-machine unit tests.
- `assets/replies/` - local static intent WAV catalog.
- `inference/best.pt` - VCM checkpoint.

## Pi Installation

Verified on the Pi:

- QuadcastRGB 1.0.5 built for ARM64.
- DuoCast USB IDs detected: `03f0:098c`, `03f0:0a8c`.
- udev permissions installed in `/etc/udev/rules.d/10-hyperx-duocast-rgb.rules`.
- User added to `hyperrgb`.
- Piper installed in `~/piper-venv`.
- Piper voice installed at `~/piper-voices/en_US-lessac-medium.onnx`.
- `openwakeword` installed in `~/piper-venv`.
- PyTorch and torchaudio installed in `~/piper-venv`.
- WILLEN is PipeWire sink 97.
- DuoCast capture device is named `HyperX DuoCast` and currently accepts 44.1 kHz.
- Static replies deployed to `~/pi5-vcm/assets/replies/`.

## Pi Commands

From a Pi shell:

```bash
cd ~/pi5-vcm

# Production combined harness
~/piper-venv/bin/python -m rpi5.run \
  --microphone \
  --checkpoint inference/best.pt \
  --device cpu \
  --wakeword alexa \
  --wakeword-threshold 0.7 \
  --input-device "HyperX DuoCast" \
  --audio-player pw-play \
  --audio-device 97 \
  --reply-dir assets/replies \
  --ack-wav assets/replies/yes.wav \
  --rgb-executable ~/.local/bin/quadcastrgb

# Wake-only diagnostic: Alexa -> yes.wav -> standby
~/piper-venv/bin/python -m rpi5.run \
  --microphone --wake-only \
  --checkpoint inference/best.pt --device cpu \
  --wakeword alexa --input-device "HyperX DuoCast" \
  --audio-player pw-play --audio-device 97 \
  --ack-wav assets/replies/yes.wav \
  --reply-dir assets/replies \
  --rgb-executable ~/.local/bin/quadcastrgb
```

Aliases in the Pi `~/.zshrc`:

- `pi5-vcm-demo` - production harness.
- `pi5-vcm-debug` - production harness plus timestamped wake score log.

The Pi shell may be Bash and may not have zsh installed; source the aliases only if zsh is available. Direct commands above are authoritative.

## Validation Completed

Local:

```bash
../pi5venv/bin/python -m unittest discover -s tests -v
../pi5venv/bin/python -m compileall -q rpi5 tests
```

Result: 4 state-machine tests passed; compilation passed.

Pi:

```bash
~/piper-venv/bin/python -m compileall -q rpi5
~/piper-venv/bin/python -m rpi5.run --self-test --checkpoint inference/best.pt --device cpu --no-warmup
```

The Pi self-test produced four structured events with CPU inference latency around 17-38 ms. The toy checkpoint classified synthetic samples as OOV; this is a model-quality/training issue, not a runtime startup failure.

Hardware checks passed:

- QuadcastRGB solid/cycle/wave/off commands.
- `pw-play --target=97` to WILLEN.
- Piper-generated WAV playback.
- 17 static reply WAVs validated and representative WAVs played.

## Known Gaps

1. `rpi5.run` is now wired to `HarnessStateMachine`, but the audio generator still owns wake/listening capture details. A later refactor should make the state-machine boundary fully explicit for wake, command timeout, and separate capture sessions.
2. Current VAD trailing silence behavior is simpler than the requested approximately 1-second policy.
3. Action dispatch remains dry-run; real GPIO/MQTT/MPD/timer integrations are not implemented.
4. Intent taxonomy in `config.py` has 15 commands plus OOV and does not exactly match the newer handover taxonomy.
5. The installed openWakeWord package provides Alexa and other models, but the requested “Hey Rhasspy” model was not available in the tested package.
6. The checkpoint is a toy/untrained model for demonstration; real dataset training remains required.
8. Avoid broad `pkill -f` commands containing their own command text; they can terminate the SSH wrapper. Use exact PID selection from `ps` when cleaning processes.

## Next Recommended Slice

1. Add explicit `CaptureSession` APIs: `wait_for_wake()`, `capture_command(timeout_s=7)`, and `reset()`.
2. Move wake-word listening and command listening into separate state-machine-controlled phases.
3. Add one-second trailing silence and seven-second command-start timeout.
4. Add action failure/UNKNOWN reply policy using `oov.wav` or a dedicated failure WAV.
5. Add tests for command timeout, wake acknowledgement isolation, action failure, and static reply selection.
6. Benchmark Pi p50/p95 wake-to-ack and wake-to-reply latency.
