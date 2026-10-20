# Raspberry Pi 5 wake runner

This folder is the self-contained source package for the Pi 5 Alexa
wake-word acknowledgement runner. It can be copied or archived on its own.
Runtime-installed files stay in this folder too: `.venv/`, `models/`, and
`.config/`. Run setup after placing the folder at its final location because
Python virtual environments contain absolute paths.

From this folder, set up the wake-word runtime with:

```bash
bash setup_pi.sh
```

The setup script creates a Python virtual environment, installs the pinned
openWakeWord release with its ONNX runtime dependencies, downloads the Alexa
and shared feature ONNX models from the upstream `v0.5.1` release, and verifies
their SHA-256 hashes. It requires a 64-bit Raspberry Pi OS, Python 3.10–3.13,
ALSA capture utilities, PipeWire, and the Raspberry Pi OS `python3-gpiozero`
and `python3-lgpio` packages. It discovers the connected **WILLEN** PipeWire
sink and stores its target name in this folder's `.config/`. Python packages
and downloaded wake models live in this folder's `.venv/` and `models/`;
the host OS supplies ALSA, PipeWire, and GPIO access. Setup also installs the
packaged WILLEN no-idle-suspend rule under
`~/.config/wireplumber/wireplumber.conf.d/` and restarts WirePlumber when that
rule changes. It verifies that both `node.pause-on-idle=false` and a zero
session suspend timeout are active on the detected WILLEN sink.

The speaker rule is a persistent WirePlumber user configuration, independent
of `run_pi.sh`; it remains active when the listener is stopped. Rerun
`bash setup_pi.sh` after copying an updated runner package to install updates to
the rule. This disables host-side idle suspension only; the speaker's own
battery-saving auto-off behavior is separate.

Start listening after setup:

```bash
bash run_pi.sh
```

To keep it running under systemd and collect timestamped logs in the journal:

```bash
bash install_service.sh
```

Check the service and follow its logs:

```bash
systemctl --user status pi5-vcm-inference.service
journalctl --user -u pi5-vcm-inference.service -f
```

The service runs continuously, rearming the wake-word model after every
acknowledgement. The systemd user service starts with the user's session. To
keep it running through logout and start automatically at boot, enable
lingering once as prompted by the installer:

```bash
sudo loginctl enable-linger "$USER"
```

Remove the systemd service without deleting the runner or its downloaded
assets:

```bash
bash uninstall_service.sh
```

The runner captures mono 16-kHz audio from the HyperX DuoCast, detects
“Alexa,” stops capture, and plays the acknowledgement beep through WILLEN.
The external GPIO indicator is disabled by default because the LED's BCM pin
has not been specified. Once wired, set it for a run, for example:

```bash
PI5_VCM_LED_GPIO=17 bash run_pi.sh
```

The foreground QuadcastRGB build, `quadcastrgb-foreground`, is automatically
used by `run_pi.sh` when present. The standard upstream binary daemonizes and
keeps the USB interface open, so it cannot be switched between warmup and wake
states by the runner. Build the foreground variant on the Pi with:

```bash
git clone --depth 1 https://github.com/Ors1mer/QuadcastRGB.git \
  "$HOME/.local/src/QuadcastRGB"
make -C "$HOME/.local/src/QuadcastRGB" dev
install -m 0755 "$HOME/.local/src/QuadcastRGB/dev" \
  "$HOME/.local/bin/quadcastrgb-foreground"
```

The Pi's DuoCast USB ID (`03f0:098c`) is supported by
[QuadcastRGB](https://github.com/Ors1mer/QuadcastRGB). When enabled, the
microphone cycles RGB during startup warmup, turns its lights off when warmup
completes, lights blue when Alexa is detected, changes to green while Piper's
reply is playing, and turns off after command handling. The GPIO indicator
remains a simple on/off indicator. If the foreground binary is not on `PATH`
or in `~/.local/bin/`, set its path explicitly:

```bash
PI5_VCM_RGB_EXECUTABLE=/path/to/quadcastrgb-foreground bash run_pi.sh
```

Set `PI5_VCM_HYPERX_RGB=0` to disable automatic RGB control, or set it to `1`
to require the utility and fail at startup if it is missing. The utility
requires USB access to the microphone; see its upstream instructions if it
reports a permissions error. A separate GPIO indicator can be used alongside
the DuoCast RGB by also setting `PI5_VCM_LED_GPIO`.

For a one-shot diagnostic rather than a continuous listener, pass `--once`:

```bash
bash run_pi.sh --once
```

By default, the runner handles wake acknowledgement only. To enable the Phase 1
command path, provide an exported ME2 ONNX model, its paired training-only mel
statistics, and the model's tuned in-scope confidence threshold:

```bash
bash run_pi.sh \
  --intent-model /path/to/model.onnx \
  --mel-stats /path/to/mel_stats.npz \
  --intent-threshold 0.78
```

The runner acknowledges the wake word, records one mono 16-kHz command (up to
8 seconds, ending after 0.8 seconds of silence), runs ME2 inference, and dispatches
an accepted intent. It waits up to 3 seconds for speech and ignores recordings
with less than 0.24 seconds of detected voice. The three `--intent-*` options
must be supplied together. An absent/unavailable action integration is reported
as `unavailable`; a rejected or out-of-scope prediction is not executed.

To test one scaffolded action without a compatible ME2 model, force an intent
for the next wake-command only:

```bash
bash run_pi.sh --force-intent TIME
```

After Alexa is acknowledged, the runner still requires a voiced command, but
does not interpret its words; it dispatches `TIME` once and clears the override
when it returns to wake standby. This is useful for exercising an action while
its model integration is unfinished. `--force-intent` cannot be combined with
the ME2 model options. Select media or GPIO intents only when you intend those
real actions to occur.

For a persistent, explicitly enabled TIME-only test mode, use:

```bash
bash run_pi.sh --dummy-time
```

Each Alexa wake prompts for a voiced command, ignores its words, dispatches
`TIME`, speaks the result with Piper, and remains enabled for later wakes.
Silence or a missed wake-command does not clear the mode. Stop the process with
Ctrl+C to return to standby; without `--dummy-time`, normal wake-only behavior
is unchanged. This is a stub at the dispatch seam, **not an ONNX model**, and
it bypasses real ME2 inference. Startup and result logs identify dummy mode.
Do not combine it with `--force-intent` or the real ME2 model options.

To leave the runner persistently testing a different intent, use
`--dummy-intent` instead. It dispatches that intent after every voiced
command, ignores the command's words, and bypasses ME2 inference:

```bash
bash run_pi.sh --dummy-intent WEATHER
```

For the systemd service, set `PI5_VCM_DUMMY_INTENT` to the uppercase intent
name when installing it, for example `PI5_VCM_DUMMY_INTENT=WEATHER
bash install_service.sh`. The service remains on that intent until explicitly
reinstalled in another mode. A one-wake `--force-intent` test does not change
the persistent service configuration.

`setup_pi.sh` installs Piper TTS and downloads the `en_US-lessac-medium` voice.
When command handling is enabled, the runner loads Piper's voice model once at
startup and reuses it for each response. Startup also runs three synthetic
silence frames through the wake-word detector and resets its state, then
synthesizes and discards a short Piper warmup WAV. Neither warmup reads from the
microphone or plays sound. Startup logs the warmup durations before declaring
the listener ready.

For real responses, Piper synthesizes to a temporary WAV, which is played
through the configured PipeWire/ALSA output and then removed. On the configured
Pi 5, a short TIME phrase took about 3.7–3.9 seconds through a fresh Piper CLI
process, versus about 0.31 seconds to synthesize with the already-loaded Piper
API. Model loading costs about 3.3 seconds once at startup instead of on every
reply. Command-end detection still waits for 0.8 seconds of silence; that pause
is retained to avoid cutting off commands. These timings are local measurements
and may vary with system load.

The selected voice model's source dataset license limits use to research
purposes; see its
[model card](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/lessac/medium/MODEL_CARD)
and [license](https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html).

For command-path latency diagnostics, the runner logs command-capture,
intent-execution, speech-synthesis, speech-playback, and total wake-to-reply
durations for each interaction. The total starts after wake acknowledgement
and includes waiting for the command, the command itself, and reply playback.

Setup does not install or select the ME2 model, mel statistics, or threshold.
Keep those artifacts paired with the checkpoint/export report that produced
them.

To keep the persistent dummy TIME mode in the systemd user service, explicitly
set the installer flag when creating or replacing the service:

```bash
PI5_VCM_DUMMY_TIME=1 bash install_service.sh
```

Use `PI5_VCM_DUMMY_INTENT=WEATHER` to persistently test WEATHER instead. Set
neither variable to restore normal standby; the service installer defaults to
dummy mode off.

## Intent action scaffold

The dispatcher accepts either a class ID (0–19) or an intent label using the
ME2 20-class order. The caller must apply the selected model's out-of-scope
threshold before dispatch. The dispatcher returns an `ExecutionResult` with
`completed`, `unavailable`, or `rejected` status; it does not treat a missing
integration as a successful action.

The currently implemented local actions are:

| Intent(s) | Action |
| --- | --- |
| `PLAY_MUSIC`, `PAUSE`, `STOP`, `NEXT` | Play supported files recursively from `~/Music` with `ffplay`; tracks are ordered by path, and `NEXT` advances and wraps. |
| `VOLUME_UP`, `VOLUME_DOWN` | Adjust the default PipeWire output by 5%, capped at 100%; this affects music and Piper speech. |
| `TIME` | Speak the Pi's local system time using Piper TTS. |
| `WEATHER` | Speak current conditions in Quezon City using OpenWeather Current Weather API. |
| `LIGHT_ON`, `LIGHT_OFF` | Toggle one GPIO Zero output if `PI5_VCM_LIGHT_GPIO` is configured. |
| `CREATE_REMINDER`, `LIST_REMINDERS` | Persist undated reminder actions locally and speak the saved list. |

`CREATE_REMINDER` stores one of the trained actions (`Drink water`, `Study`,
or `Exercise`) with a local creation timestamp. The model slot does not
contain a due date or time, so these are persistent list entries, not
scheduled notifications. If CREATE_REMINDER is forced without an inferred
slot, the runner stores a generic `Reminder`. `LIST_REMINDERS` speaks at most
the five newest items and reports how many additional entries remain.

By default, reminders are stored in
`~/.config/pi5-vcm/reminders.json`. Set `PI5_VCM_REMINDERS_FILE` in the
service's private environment file to override the location. Writes replace
the JSON file atomically; malformed or inaccessible store data is reported as
an error rather than treated as an empty list.

Weather reads the API key from `OPENWEATHER_API_KEY` and queries
`Quezon City,PH` through OpenWeather's
[Current Weather API](https://openweathermap.org/api/current?collection=current_forecast)
using metric units. This uses the city's query name directly, avoiding a
separate geocoding request. The response's `main.temp` and
`weather[0].description` are spoken. Temperature is phrased as metric
"degrees" to avoid an unnecessary unit word in speech. The runner does not
log the key or request URL.

For the systemd service, create a private environment file on the Pi:

```bash
mkdir -p "$HOME/.config/pi5-vcm"
install -m 600 /dev/null "$HOME/.config/pi5-vcm/environment"
nano "$HOME/.config/pi5-vcm/environment"
```

Add this line to the file, replacing the placeholder locally (do not commit
the file or share the key in chat):

```text
OPENWEATHER_API_KEY=your_openweather_api_key
```

Then install or refresh the service so it loads the environment file:

```bash
PI5_VCM_DUMMY_INTENT=WEATHER bash install_service.sh
```

The installer includes the file as an optional systemd `EnvironmentFile`; if
it is absent, other runner features continue to work and WEATHER reports that
weather configuration is unavailable. The dummy intent setting makes the
Pi speak live WEATHER after every voiced command; change the installer mode
explicitly when switching to another intent or back to normal inference.

The local music player uses `ffplay` from FFmpeg and scans `~/Music`
recursively for AAC, AIFF, FLAC, M4A, MP3, Ogg, Opus, and WAV files. Tracks
are played in case-insensitive path order; `NEXT` wraps to the first track.
The collection is rescanned for each play or next action, so added or removed
files are picked up without restarting the service.
`PLAY_MUSIC` resumes a paused track, `PAUSE` pauses the current process,
`STOP` terminates it, and `NEXT` starts the following track. The process is
stopped when the inference service exits. Volume intents use PipeWire's
`wpctl` to adjust the default output in 5% steps, affecting all output
including spoken replies; adjustments are capped at 100%.

For light control, wire the selected BCM GPIO to a suitable relay/driver and
configure its pin, for example `PI5_VCM_LIGHT_GPIO=17`. Do not connect a relay
coil or other high-load device directly to a Raspberry Pi GPIO. GPIO actions
require `gpiozero`, which is supplied by the Raspberry Pi OS packages
installed by `setup_pi.sh`.

Telephony, messaging, timers, alarms, temperature, brightness, and color
actions return `unavailable` until their corresponding service or hardware
adapter is configured. `OUT_OF_SCOPE` returns `rejected`. Timer, alarm,
temperature, brightness, and color slot values are not yet acted on.

An inference loop can dispatch a predicted class ID like this:

```python
from intent_execution import IntentDispatcher

dispatcher = IntentDispatcher()
try:
    result = dispatcher.execute(predicted_intent_id)
    print(result.status, result.message)
finally:
    dispatcher.close()
```

The model's six slot outputs are decoded. CREATE_REMINDER uses its predicted
action slot; timer, alarm, temperature, brightness, and color slot actions
remain unimplemented. The wake-word model files downloaded by setup are stored
under `models/openwakeword/`; they are not embedded into the Git repository.

## Dependency note

openWakeWord 0.6.0 declares `tflite-runtime` as a Linux dependency, which does
not currently provide a compatible wheel for this Pi's Python 3.13 setup. The
installer skips package dependency resolution for openWakeWord, installs its
ONNX runtime dependencies explicitly, and uses only ONNX models. This avoids
installing the unused TFLite backend.
