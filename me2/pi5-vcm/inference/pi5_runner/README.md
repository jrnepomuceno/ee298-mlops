# Raspberry Pi 5 wake runner

This folder contains the deployable inference runner package for 64-bit
Raspberry Pi OS. It can be packaged on a development machine and installed on
a compatible Pi without copying the rest of the repository. The install flow
creates the Python environment, downloads runtime assets, and installs the
user service. Machine-specific files stay beside the runner: `.venv/`,
`models/`, and `.config/`.

## Build and install a deployment package

From this folder on a development machine with `tar` and a SHA-256 utility:

```bash
bash package_pi.sh
```

This creates `dist/pi5-vcm-inference-<version>.tar.gz` and a matching
`.sha256` checksum. The archive contains the runner source and scripts, but
excludes Python environments, downloaded models, local configuration, and
Python caches. To choose an output directory or version:

```bash
bash package_pi.sh /path/to/output v1.0.0
```

Copy the archive to a compatible Pi, then install it under the current user's
home directory:

```bash
mkdir -p "$HOME/.local/opt/pi5-vcm"
tar -xzf pi5-vcm-inference-<version>.tar.gz \
  -C "$HOME/.local/opt/pi5-vcm" --strip-components=1
cd "$HOME/.local/opt/pi5-vcm"
bash install_pi.sh
```

`install_pi.sh` runs `setup_pi.sh`, then enables and starts the systemd user
service. Use `bash install_pi.sh --setup-only` to prepare runtime dependencies
and assets without starting the service; start manually with `bash run_pi.sh`
or install the service later with `bash install_service.sh`.

The package currently targets 64-bit Raspberry Pi OS (`aarch64`), Python
3.10–3.13, and ALSA. It uses the system's default ALSA capture and playback
devices unless you configure overrides. Install host packages before running
the installer:

```bash
sudo apt install alsa-utils curl
```

Connect and configure a microphone and speaker visible to ALSA. To select
specific devices, set `PI5_VCM_INPUT_DEVICE` to an ALSA capture device and
`PI5_VCM_OUTPUT_DEVICE` to an ALSA playback device or `pipewire:<target>` in
`~/.config/pi5-vcm/environment`. Example:

```text
PI5_VCM_INPUT_DEVICE=plughw:1,0
PI5_VCM_OUTPUT_DEVICE=pipewire:auto
```

PipeWire is optional; install `pipewire`, `wireplumber`, and `pipewire-bin` if
you want to use PipeWire output routing. The legacy `PI5_VCM_PIPEWIRE_TARGET`
setting and an existing `.config/pipewire-target` file are also honored for
upgrades. GPIO packages are optional and only needed for GPIO indicator or
light intents. DuoCast RGB/color/brightness control remains an optional
hardware-specific feature and is disabled when its utility is absent. The
inference runner itself does not require a particular microphone or speaker.

For an update, stop the user service, extract the newer archive over this
installation, then rerun setup and service installation:

```bash
systemctl --user stop pi5-vcm-inference.service
tar -xzf pi5-vcm-inference-<version>.tar.gz \
  -C "$HOME/.local/opt/pi5-vcm" --strip-components=1
cd "$HOME/.local/opt/pi5-vcm"
bash install_pi.sh
```

The archive intentionally excludes existing local runtime configuration,
models, and the virtual environment. Setup reuses/downloads assets as needed.

From this folder, set up the wake-word runtime with:

```bash
bash setup_pi.sh
```

The setup script creates a Python virtual environment, installs the pinned
The setup script bootstraps pinned `uv` 0.12.19 into `.tools/` when the matching
version is not already on `PATH`, creates a Python virtual environment, and
installs the locked dependencies with `uv sync`. The lockfile excludes only
OpenWakeWord's unused `tflite-runtime` dependency; this runner uses its ONNX
backend. Setup then downloads the Alexa and shared feature ONNX models from
upstream `v0.5.1` and verifies their SHA-256 hashes. It requires 64-bit
Raspberry Pi OS, Python 3.10–3.13, ALSA capture/playback utilities, and `curl`.
Python packages and downloaded wake models live in this folder's `.venv/` and
`models/`; ALSA drivers and optional audio/GPIO services are supplied by the
host OS. Setup does not scan for, configure, or require a specific microphone,
speaker, PipeWire server, or GPIO package.

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

The runner captures mono 16-kHz audio from the ALSA system default, detects
“Alexa,” stops capture, and plays the acknowledgement beep through the ALSA
system default. Override the input and output with `PI5_VCM_INPUT_DEVICE` and
`PI5_VCM_OUTPUT_DEVICE` or the corresponding command-line options. The
external GPIO indicator is disabled by default because the LED's BCM pin has
not been specified. Once wired, set it for a run, for example:

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
completes, shows the selected interaction color when Alexa is detected, and
pulses that color while Piper's reply is playing. `COLOR` changes this
preference to Red, Blue, or Green and stores it across service restarts at
`~/.config/pi5-vcm/interaction-color.json` (override with
`PI5_VCM_INTERACTION_COLOR_FILE`). It defaults to Blue. The preference affects
the DuoCast only; the optional GPIO indicator remains on/off. Standby remains
off and startup warmup continues cycling. If the foreground binary is not on
`PATH` or in `~/.local/bin/`, set its path explicitly:

```bash
PI5_VCM_RGB_EXECUTABLE=/path/to/quadcastrgb-foreground bash run_pi.sh
```

`BRIGHTNESS` sets the DuoCast RGB output to the trained level of 20%, 60%, or
100%. The selected level applies to subsequent wake acknowledgements and
spoken-reply pulses for the current runner process; it resets to 100% after a
service restart. It requires DuoCast RGB control to be enabled. It does not
change the optional external GPIO indicator.

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
Each real inference also emits a benchmark-compatible key/value record, for
example `benchmark event: intent=TIMER slot=30 seconds infer_ms=42.1
audio_ms=1280.0`. `infer_ms` covers the prediction call and `audio_ms` is the
duration of the recorded command audio. Fixed intents and out-of-scope
predictions omit `slot`; existing human-readable prediction and execution logs
remain unchanged.

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
threshold before dispatch. The dispatcher returns an `ExecutionResult` with `completed`, `submitted`,
`unavailable`, or `rejected` status; it does not treat a missing integration as
a successful action. `submitted` means an external provider accepted a request
for processing; it does not imply the resulting action or delivery completed.

The currently implemented local actions are:

| Intent(s) | Action |
| --- | --- |
| `PLAY_MUSIC`, `PAUSE`, `STOP`, `NEXT` | Play supported files recursively from `~/Music` with `ffplay`; tracks are ordered by path, and `NEXT` advances and wraps. |
| `VOLUME_UP`, `VOLUME_DOWN` | Adjust the default PipeWire output by 5%, capped at 100%; this affects music and Piper speech. |
| `TIME` | Speak the Pi's local system time using Piper TTS. |
| `WEATHER` | Speak current conditions in Quezon City using OpenWeather Current Weather API. |
| `LIGHT_ON`, `LIGHT_OFF` | Toggle one GPIO Zero output if `PI5_VCM_LIGHT_GPIO` is configured. |
| `TIMER` | Start a local countdown for 10 seconds, 30 seconds, or 1 minute and speak when it expires. |
| `ALARM` | Set a one-shot alarm for 6:00 AM, 8:00 AM, or 9:00 PM local device time. |
| `TEMPERATURE` | Speak confirmation for the selected 18, 22, or 26 degree set point. |
| `BRIGHTNESS` | Set DuoCast RGB to the selected 20%, 60%, or 100% level when RGB control is enabled. |
| `COLOR` | Set the persisted DuoCast interaction color to Red, Blue, or Green. |
| `CREATE_REMINDER`, `LIST_REMINDERS` | Persist undated reminder actions locally and speak the saved list. |
| `CALL` | Dial the Baresip current contact when the opt-in local control integration is enabled. |
| `MESSAGE` | Submit the configured fixed message text to the Baresip current contact. |

The TIMER slot selects one of the three trained durations. Only one countdown
is active at a time; setting another replaces the current timer. It runs
asynchronously, is not persisted across service restarts, and speaks “Your
timer is up.” on expiry. A direct or forced TIMER dispatch without an inferred
slot defaults to 10 seconds.

The ALARM slot selects one of the three trained local clock times. The alarm
fires at the next occurrence of that time (tomorrow if today's time has
passed), replaces any existing alarm, and speaks the scheduled time when it
fires. It is independent of TIMER and is not persisted across service
restarts. A direct or forced ALARM dispatch without an inferred slot defaults
to 6:00 AM. There is no model-supported cancel-alarm intent.

The TEMPERATURE slot selects a set point (18, 22, or 26 degrees). The runner
speaks “Setting the temperature to {value} degrees.” only; no thermostat
controller or temperature sensor is configured, so it does not change or
measure physical temperature. A direct or forced TEMPERATURE dispatch without
an inferred slot asks the user to choose one of the supported set points.

`COLOR` changes the DuoCast interaction color: wake acknowledgement shows the
selected color solid, and spoken replies pulse in that color. The selection is
saved immediately and reused after service restarts. It does not change
startup cycling or standby-off behavior. The optional GPIO indicator remains
on/off only. A COLOR confirmation is spoken while the new color pulses when
DuoCast RGB is enabled.

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

For an optional external status LED, connect a low-current LED through a
series resistor between a supported BCM GPIO and ground. For example, BCM
GPIO17 is physical header pin 11; a ground connection is physical pin 6.
Configure it for a one-off run with `PI5_VCM_LED_GPIO=17 bash run_pi.sh`.
For the systemd service, put `PI5_VCM_LED_GPIO=17` in the private environment
file before reinstalling the service. The GPIO is only an on/off indicator;
it is separate from DuoCast brightness and from the `PI5_VCM_LIGHT_GPIO`
output used by `LIGHT_ON`/`LIGHT_OFF`. Use BCM numbering for configuration,
not physical header pin numbers.

`CALL` uses Baresip's `contact.so` commands through its JSON-netstring
`ctrl_tcp` module. MESSAGE uses Baresip's `-e` command-line option to submit
`/message <text>` to the configured current contact. Configure Baresip on the Pi
before enabling the integration:

```text
module alsa.so
module_app contact.so
module_app ctrl_tcp.so
audio_source alsa,default
ctrl_tcp_listen 127.0.0.1:4444
```

Keep the listener on loopback; do not expose the control port on a LAN
interface. `audio_source` must point to the same ALSA capture device used by
the runner (`PI5_VCM_INPUT_DEVICE`, default `default`); replace `default` above
if a device-specific ALSA name is configured. This lets Baresip acquire the
configured microphone for the SIP call. Baresip must have a current contact
selected. Its existing account configuration remains in `~/.baresip/accounts`.

To have the inference runner start Baresip in the background and stop the
process it started when the runner exits, put these lines in the private
`~/.config/pi5-vcm/environment` file:

```text
PI5_VCM_BARESIP_ENABLED=1
PI5_VCM_MESSAGE_TEXT="Demo message configured privately"
```

The integration is disabled by default. With it enabled, CALL starts or
connects to Baresip, validates the module and loopback-listener configuration,
and keeps one control connection open for call-state events. CALL dials
Baresip's current contact. The runner stops its `arecord` capture before
dialing, so it releases the shared microphone for Baresip; it does not capture
VCM commands during the call. Once Baresip reports that the call closed, the
runner resumes wake-word capture. The called device can end the demo call.
MESSAGE runs `baresip -f ~/.baresip -e "/message <configured text>" -t 8` as a
one-shot command. If the runner owns a background Baresip process, it stops that
process for the command and restarts it afterward; it refuses to start a second
instance when connected to an externally managed process. Message text is fixed
by `PI5_VCM_MESSAGE_TEXT`, not transcribed or generated from command audio.
Baresip's successful local command does not confirm SIP server acceptance or
recipient delivery, so the intent reports `submitted` with delivery explicitly
unconfirmed. If Baresip or the message text is not configured, the intent
reports `unavailable`. Tests mock the process invocation and control socket;
they do not start Baresip or send calls/messages.

`OUT_OF_SCOPE` returns `rejected`. The BRIGHTNESS slot selects 20%, 60%, or
100%; TIMER's duration, ALARM's time, TEMPERATURE's set point, and COLOR's
named color slots are also implemented. TEMPERATURE speaks a set-point
confirmation without controlling hardware.

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

The model's six slot outputs are decoded. TIMER uses its predicted duration
slot, ALARM uses its predicted local clock-time slot, TEMPERATURE speaks its
predicted set-point slot, BRIGHTNESS uses its predicted level slot, COLOR uses
its predicted named-color slot, and CREATE_REMINDER uses its predicted action
slot. The wake-word model files downloaded by setup are stored under
`models/openwakeword/`; they are not embedded into the Git repository.

## Dependency note

openWakeWord 0.6.0 declares `tflite-runtime` as a Linux dependency, which does
not currently provide a compatible wheel for this Pi's Python 3.13 setup. The
installer skips package dependency resolution for openWakeWord, installs its
ONNX runtime dependencies explicitly, and uses only ONNX models. This avoids
installing the unused TFLite backend.
