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
The GPIO indicator is disabled by default because the LED's BCM pin has not
been specified. Once wired, set it for a run, for example:

```bash
PI5_VCM_LED_GPIO=17 bash run_pi.sh
```

For a one-shot diagnostic rather than a continuous listener, pass `--once`:

```bash
bash run_pi.sh --once
```

This is still only the wake-and-acknowledgement slice. It does not record a
command, run ME2 inference, or execute actions. The accepted ME2 model and its
normalization/configuration are intentionally not included yet. The model
files downloaded by setup are stored under `models/openwakeword/`; they are
not embedded into the Git repository.

## Dependency note

openWakeWord 0.6.0 declares `tflite-runtime` as a Linux dependency, which does
not currently provide a compatible wheel for this Pi's Python 3.13 setup. The
installer skips package dependency resolution for openWakeWord, installs its
ONNX runtime dependencies explicitly, and uses only ONNX models. This avoids
installing the unused TFLite backend.
