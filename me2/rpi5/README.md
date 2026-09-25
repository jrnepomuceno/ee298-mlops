# Pi 5 runtime harness

This is the first Pi-local mini-Alexa runtime slice. It reuses the model
contract in `inference/infer.py`, emits one structured event per utterance,
produces a deterministic offline reply, and dispatches only allowlisted dry-run
actions. It has no ASR, network, GPIO, MQTT, or LLM dependency.

From the project root:

```bash
./rpi5/run_pi5_harness.sh --self-test --checkpoint inference/best.pt
python -m rpi5.run --self-test --checkpoint inference/best.pt
python -m rpi5.run --input command.wav --checkpoint inference/best.pt
python -m rpi5.run --microphone --checkpoint inference/best.pt
python -m rpi5.run --microphone --wakeword alexa --checkpoint inference/best.pt
python -m rpi5.run --microphone --wakeword alexa \
	--rgb-executable /usr/local/bin/quadcastrgb \
	--checkpoint inference/best.pt
python -m rpi5.run --microphone --wakeword alexa \
	--tts-model voices/en_US-lessac-medium.onnx \
	--audio-device plughw:1,0 \
	--rgb-executable /home/jdrnepomuceno9/.local/bin/quadcastrgb \
	--checkpoint inference/best.pt
python -m rpi5.mcp_server --checkpoint inference/best.pt
```

Microphone mode listens on the default 16 kHz input device. The energy VAD
starts after two loud frames, keeps a short pre-roll, and ends after roughly
360 ms of silence or five seconds. Press `Ctrl-C` to stop. Install
`requirements-runtime.txt` first; replay and self-test do not require a
microphone.

The event contains the request ID, timestamp, input source, model result,
action result, and a `reply.text` suitable for display or offline TTS. Actions
report `side_effects: false` and are rejected for OOV or low-confidence
predictions. The VCM is the command recognizer; it does not transcribe open
speech like an ASR system.

With a wake word enabled, the interaction is two-stage: say `Alexa`, the
device replies `Yes?`, then speak the command. Wake-word audio and the `Yes?`
reply are discarded before command VAD begins, preventing the assistant from
recognizing its own acknowledgement as a command.

The actual harness microphone path also uses static intent WAVs. Run it with
`--reply-dir assets/replies`; it selects `<intent>.wav` after inference and
never starts Piper for command replies.

Fixed intent replies are the pre-recorded WAVs in `assets/replies/<intent>.wav`,
played back fully offline. These fixed files confirm the action but do not
include variable slot values; slot-aware reply text is built by `replies.py`.

Wake-word support is optional and uses a pretrained `openwakeword` model.
Install `requirements-wakeword.txt` before using `--wakeword alexa`. The exact
pretrained model name must be available in the installed openWakeWord release.

## Wireless speaker and TTS

Install Piper with `requirements-tts.txt`, download a Piper `.onnx` voice model
and its matching `.json` file, and configure the wireless speaker as an ALSA
output. `--tts-model` enables spoken replies; without it, replies remain
text-only. Playback blocks until `aplay` finishes, so the RGB cycle remains
active during speech and turns off afterward.

Use `aplay -l` to find an output device. Pass an ALSA device such as
`plughw:1,0` with `--audio-device`, or omit it to use the system default.

## DuoCast RGB feedback

`--rgb-executable` enables the optional QuadcastRGB controller. The harness
sequence is black (`solid 000000`) at idle, a fast cyan wave (`-s 10 wave
00A0FF`) after wake-word detection, a slower cyan wave during processing, and a
green wave while TTS is playing. After TTS, it waits 100 ms and returns to
black. The microphone remains active throughout; only the light controller is
changed. Without this option, RGB commands are dry-run only.

## MCP on the Pi

The MCP server loads the harness once and exposes `health`, `recognize_file`,
and `listen_once` over MCP stdio. An MCP client running on the Pi can call it
without internet access. The tools return the same event, including the
deterministic `reply.text` intended for display or offline TTS.
