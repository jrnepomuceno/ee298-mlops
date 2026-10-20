#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
ASSET_DIR="$SCRIPT_DIR/models/openwakeword"
TTS_ASSET_DIR="$SCRIPT_DIR/models/tts"
PYTHON_BIN="${PYTHON:-python3}"
CONFIG_DIR="$SCRIPT_DIR/.config"

fail() {
    printf 'Pi 5 inference setup failed: %s\n' "$*" >&2
    exit 1
}

[[ "$(uname -m)" == "aarch64" ]] || fail "this setup currently targets 64-bit Raspberry Pi OS (aarch64)"
command -v arecord >/dev/null || fail "install ALSA utilities first (sudo apt install alsa-utils)"
command -v aplay >/dev/null || fail "install ALSA utilities first (sudo apt install alsa-utils)"
command -v curl >/dev/null || fail "curl is required to fetch the pinned wake-word assets"
command -v sha256sum >/dev/null || fail "sha256sum is required to verify downloaded assets"

"$PYTHON_BIN" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)' \
    || fail "Python 3.10–3.13 is required; detected $("$PYTHON_BIN" --version 2>&1)"

if ! "$PYTHON_BIN" -m venv --help >/dev/null 2>&1; then
    fail "Python venv support is missing; install it with sudo apt install python3-venv"
fi

mkdir -p "$(dirname -- "$VENV_DIR")" "$ASSET_DIR" "$CONFIG_DIR"
mkdir -p "$TTS_ASSET_DIR"
"$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
export PYTHONNOUSERSITE=1

"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check --upgrade pip

# openWakeWord declares tflite-runtime as a Linux dependency, but this runtime
# uses its ONNX backend; install only the dependencies required for that path.
"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check \
    --ignore-installed \
    "numpy>=1.23,<3" \
    "onnxruntime>=1.10,<2" \
    "scipy>=1.3,<2" \
    "scikit-learn>=1,<2" \
    "requests>=2,<3" \
    "tqdm>=4,<5"
"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check \
    --no-deps "openwakeword==0.6.0"
"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check \
    "piper-tts==1.8.0"
"$VENV_DIR/bin/python" -m piper.download_voices \
    --data-dir "$TTS_ASSET_DIR" en_US-lessac-medium
[[ -s "$TTS_ASSET_DIR/en_US-lessac-medium.onnx" ]] \
    || fail "Piper voice model download is missing"
[[ -s "$TTS_ASSET_DIR/en_US-lessac-medium.onnx.json" ]] \
    || fail "Piper voice configuration download is missing"

download_asset() {
    local filename="$1"
    local expected_sha256="$2"
    local url="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/$filename"
    local target="$ASSET_DIR/$filename"
    local temporary

    if [[ -f "$target" ]] && printf '%s  %s\n' "$expected_sha256" "$target" | sha256sum --check --status; then
        printf 'Verified existing asset: %s\n' "$filename"
        return
    fi

    temporary="$(mktemp "$ASSET_DIR/.${filename}.XXXXXX")"
    if ! curl --fail --location --silent --show-error --retry 3 "$url" --output "$temporary"; then
        rm -f -- "$temporary"
        fail "could not download $filename from the pinned openWakeWord release"
    fi
    if ! printf '%s  %s\n' "$expected_sha256" "$temporary" | sha256sum --check --status; then
        rm -f -- "$temporary"
        fail "SHA-256 verification failed for $filename"
    fi
    mv -- "$temporary" "$target"
    printf 'Downloaded and verified: %s\n' "$filename"
}

download_asset "alexa_v0.1.onnx" \
    "6ff566a01d12670e8d9e3c59da32651db1575d17272a601b7f8a39283dfbae3e"
download_asset "embedding_model.onnx" \
    "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f"
download_asset "melspectrogram.onnx" \
    "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f"

PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$VENV_DIR/bin/python" - "$ASSET_DIR/alexa_v0.1.onnx" <<'PY'
import sys
from pathlib import Path

from wake_ack import FRAME_BYTES, OpenWakeWordDetector

detector = OpenWakeWordDetector(Path(sys.argv[1]))
score = detector.score_frame(bytes(FRAME_BYTES))
print(f"ONNX wake-word model initialized; smoke-test score={score:.4f}")
PY

printf '\nSetup complete.\n'
printf 'Start Alexa wake detection with:\n  bash "%s/run_pi.sh"\n' "$SCRIPT_DIR"
printf 'No LED is enabled until a BCM pin is supplied.\n'
printf 'Piper TTS and the en_US-lessac-medium voice are installed for spoken action responses.\n'
printf 'ME2 inference is optional: pass --intent-model, --mel-stats, and --intent-threshold; use --force-intent INTENT for a one-wake action test.\n'
printf 'Persistent TIME-only test mode is opt-in: pass --dummy-time to run_pi.sh or PI5_VCM_DUMMY_TIME=1 to install_service.sh.\n'
printf 'Audio defaults to the ALSA system devices; set PI5_VCM_INPUT_DEVICE and PI5_VCM_OUTPUT_DEVICE to override them.\n'
