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

hyperx_rgb="${PI5_VCM_HYPERX_RGB:-auto}"
if [[ "$hyperx_rgb" != "auto" && "$hyperx_rgb" != "0" && "$hyperx_rgb" != "1" ]]; then
    printf 'PI5_VCM_HYPERX_RGB must be auto, 0, or 1.\n' >&2
    exit 1
fi
rgb_executable="${PI5_VCM_RGB_EXECUTABLE:-}"
if [[ -n "$rgb_executable" && ! -x "$rgb_executable" ]]; then
    printf 'PI5_VCM_RGB_EXECUTABLE is not executable: %s\n' "$rgb_executable" >&2
    exit 1
fi
if [[ -z "$rgb_executable" ]]; then
    rgb_executable="$(command -v quadcastrgb-foreground || true)"
fi
if [[ -z "$rgb_executable" && -x "$HOME/.local/bin/quadcastrgb-foreground" ]]; then
    rgb_executable="$HOME/.local/bin/quadcastrgb-foreground"
fi
if [[ "$hyperx_rgb" != "0" && -n "$rgb_executable" && -x "$rgb_executable" ]]; then
    args+=(--hyperx-rgb-executable "$rgb_executable")
elif [[ "$hyperx_rgb" == "1" ]]; then
    printf 'HyperX RGB was required but foreground QuadcastRGB is not executable; install it or set PI5_VCM_RGB_EXECUTABLE.\n' >&2
    exit 1
fi

cd "$SCRIPT_DIR"
export PYTHONNOUSERSITE=1
exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/wake_ack.py" "${args[@]}" "$@"
