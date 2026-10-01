# Pi 5 runtime harness

This is the first Pi-local mini-Alexa runtime slice. It reuses the model
contract in `inference/infer.py`, emits one structured event per utterance,
produces a deterministic offline reply, and dispatches only allowlisted dry-run
actions. It has no ASR, network, GPIO, MQTT, or LLM dependency.

From the project root:

```bash
./rpi5/run_pi5_harness.sh --self-test --checkpoint inference/best.pt
python -m rpi5.run --self-test --checkpoint inference/best.pt
python -m rpi5.run --self-test \
	--checkpoint ~/MyProjects/intent_v1/model_int8.onnx \
	--intent-labels new_training/intent_v1_labels.json --no-warmup
python -m rpi5.run --self-test \
	--checkpoint models/onnx/v6_15m/model_int8.onnx \
	--intent-labels models/onnx/v6_15m/contract.json \
	--enable-v6-actions --no-warmup --no-volume
python -m rpi5.run --vcm-only \
	--checkpoint models/onnx/v6_15m/model_int8.onnx \
	--intent-labels models/onnx/v6_15m/contract.json \
	--enable-v6-actions --live-media --no-warmup --no-volume
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
```

## Model-free intent scenarios

The `v6_15m` export has an intent head and eight bounded slot heads. Its
contract must be supplied with `--intent-labels`. Without `--enable-v6-actions`,
the harness stays diagnostics-only. The opt-in demo action mode currently
allows the project-supported v6 intents: `TIME`, `WEATHER`, `LIST_REMINDERS`,
`LIGHT_ON`, `LIGHT_OFF`, `BRIGHTNESS`, `TEMPERATURE`, `PLAY_MUSIC`, `PAUSE`,
`STOP`, `VOLUME_UP`, `VOLUME_DOWN`, `TIMER`, `ALARM`, and `CREATE_REMINDER`.
`NEXT` remains deferred; `CALL` is rejected because v6 has no contact slot;
`MESSAGE` has no runtime handler, and v6 has no separate `stop_timer` label.
Volume thresholds are configurable with `--volume-up-threshold` and
`--volume-down-threshold`; the Pi Bash demo starts at 0.90 and 0.60 respectively.
Other intents use `--min-confidence`, which defaults to 0.70.
When one volume label wins the raw softmax, a qualifying alternative may
replace it; if neither clears its threshold, the volume pair is rejected.
Result objects include `volume_intent_scores` with the raw UP/DOWN probabilities.
`--self-test` exercises
loading and routing without live device effects. `--vcm-only` enables the
microphone pipeline; add `--live-media` to control actual playback.

Confidence defaults are read at process startup from the project-root
`confidence_thresholds.json`: `intent.default`, `intent.volume_up`,
`intent.volume_down`, `slot`, and `wakeword`. Edit that file on the Pi to tune
all demo aliases; restart the demo to apply changes. Intent, volume, and
wakeword CLI options override the file for one run. Slot confidence currently
uses the file value directly.

Before an ONNX model is available, exercise every intent through the real
facade, dry-run task executors, event generation, and reply formatting:

```bash
python -m rpi5.mock_run --intent all
python -m rpi5.mock_run --intent set_timer
python -m rpi5.mock_run --intent turn_on_lights --confidence 0.2
python -m rpi5.mock_run --intent set_temperature --temperature 25
```

The runner supplies valid deterministic slots for each supported intent plus
an OOV case. With no live flags, all tasks stay dry-run and RGB/speech are
recorded as stubs. Temperature mock values are selected with `--temperature`
(10-35 °C; default 22). Select one intent to opt into a specific physical/service
path:

```bash
python -m rpi5.mock_run --intent what_time --live-speaker \
	--piper-bin ~/piper-venv/bin/piper \
	--piper-model ~/piper-voices/en_US-lessac-medium.onnx --audio-player aplay
python -m rpi5.mock_run --intent turn_on_lights --live-lights
python -m rpi5.mock_run --intent volume_up --live-volume
python -m rpi5.mock_run --intent volume_down --live-volume
python -m rpi5.mock_run --intent what_weather --live-weather --weather-location "Quezon City"
```

For a three-second internal countdown, audible confirmation, and a persistent
ringing alarm:

```bash
python -m rpi5.mock_run --intent set_timer --live-timer --live-speaker \
	--timer-seconds 3 --ring-seconds 3 \
	--piper-bin ~/piper-venv/bin/piper \
	--piper-model ~/piper-voices/en_US-lessac-medium.onnx --audio-player pw-play
```

The mock runner uses the real process-local `TimerManager` and `TimerAlarm`:
after expiry it speaks “Your timer is up.” immediately and every 10 seconds,
while looping `timer_ding.wav` until `stop_timer` is received. In this test
runner, `--ring-seconds 3` is an observation window, after which it injects a
mock `stop_timer` intent to stop the loop; in the production `rpi5.run` path,
the loop remains active until the model recognizes `stop_timer`. Set
`--ring-seconds 12` to verify the 10-second spoken repeat in the mock test. To
test cancellation before expiry instead, use `--intent stop_timer --live-timer
--live-speaker` with the same duration; the runner arms a short timer, cancels
it, and verifies no expiry sound occurs. Test timers are process-local and do
not cancel timers in a separately running assistant.

Each live volume test snapshots the current master output, steps it by 10%,
reports the stepped value, and restores the original level in cleanup.

For a real microphone-to-speaker integration with mocked recognition, capture
one utterance and force a 25 °C temperature reply:

```bash
python -m rpi5.mock_run --intent set_temperature --temperature 25 \
	--microphone --input-device "HyperX DuoCast" --live-speaker \
	--piper-bin ~/piper-venv/bin/piper \
	--piper-model ~/piper-voices/en_US-lessac-medium.onnx --audio-player pw-play
```

Speak any short phrase after the prompt. Audio is captured by VAD but its
content is deliberately ignored; the selected mock intent and temperature
drive the task/reply. This verifies microphone capture, audio-only HVAC routing,
Piper synthesis, and speaker playback, not model recognition. WAV files are
played directly without a digital-silence pre-roll.

The speaker option synthesizes and plays the reply; the lights option uses the
selected light driver; the weather option performs a real OpenWeatherMap
request using `OPENWEATHER_API_KEY`. `--intent all` cannot be combined with a
live option. A SIP call is separately guarded: `--live-call` requires
`--intent call`, `--confirm-call`, a numeric `--mock-contact`, and a configured
available Baresip endpoint. It stays open for 15 seconds by default; set
`--call-seconds` to a value from 1 to 120. Use only a test account/number you control. The
mock runner does not open a microphone unless `--microphone` is explicitly
provided; recognition is always forced to the selected deterministic intent.

The automated coverage is `python -m unittest discover -s tests -p 'test_mock_intents.py'`;
existing peripheral unit tests separately use fake drivers and subprocesses.

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
