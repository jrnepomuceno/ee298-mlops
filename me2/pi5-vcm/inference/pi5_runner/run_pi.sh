#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
ASSET_DIR="$SCRIPT_DIR/models/openwakeword"
CONFIG_DIR="$SCRIPT_DIR/.config"
MODEL="$ASSET_DIR/alexa_v0.1.onnx"

[[ -x "$VENV_DIR/bin/python" ]] || {
    printf 'Inference environment is missing; run bash "%s/setup_pi.sh" first.\n' "$SCRIPT_DIR" >&2
    exit 1
}
[[ -f "$MODEL" ]] || {
    printf 'Alexa model asset is missing; run bash "%s/setup_pi.sh" first.\n' "$SCRIPT_DIR" >&2
    exit 1
}
[[ -f "$ASSET_DIR/embedding_model.onnx" && -f "$ASSET_DIR/melspectrogram.onnx" ]] || {
    printf 'openWakeWord feature assets are missing; run bash "%s/setup_pi.sh" first.\n' "$SCRIPT_DIR" >&2
    exit 1
}

pipewire_target="${PI5_VCM_PIPEWIRE_TARGET:-auto}"
if [[ -f "$CONFIG_DIR/pipewire-target" ]]; then
    pipewire_target="$(<"$CONFIG_DIR/pipewire-target")"
fi

args=(
    --wakeword-model "$MODEL"
    --input-device "${PI5_VCM_INPUT_DEVICE:-default:CARD=DuoCast}"
    --output-device "pipewire:$pipewire_target"
)
if [[ -n "${PI5_VCM_LED_GPIO:-}" ]]; then
    args+=(--led-gpio "$PI5_VCM_LED_GPIO")
else
    args+=(--no-led)
fi

cd "$SCRIPT_DIR"
export PYTHONNOUSERSITE=1
exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/wake_ack.py" "${args[@]}" "$@"
