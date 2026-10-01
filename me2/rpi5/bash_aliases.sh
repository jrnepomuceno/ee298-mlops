#!/usr/bin/env bash

_pi5_v6_demo() {
    local repo="$HOME/MyProjects/ee298-mlops/me2"
    cd "$repo" || return

    set -a
    if [[ -f "$HOME/.config/vcm.env" ]]; then
        source "$HOME/.config/vcm.env"
    fi
    set +a
    mkdir -p logs

    exec "$HOME/piper-venv/bin/python" -m rpi5.run "$@" \
        --checkpoint models/onnx/v6_15m/model_int8.onnx \
        --intent-labels models/onnx/v6_15m/contract.json \
        --enable-v6-actions --live-media --device cpu \
        --audio-player pw-play --audio-device WILLEN \
        --reply-dir "$repo/assets/replies" \
        --ack-wav "$repo/assets/replies/ack_beep.wav" \
        --rgb-executable "$HOME/.local/bin/quadcastrgb"
}

    alias pi5-v6-demo='_pi5_v6_demo --microphone --wakeword alexa'
    alias pi5-v6-debug='_pi5_v6_demo --microphone --wakeword alexa --wakeword-log "$HOME/MyProjects/ee298-mlops/me2/logs/v6-demo-wake.log"'
alias pi5-v6-vcm-only='_pi5_v6_demo --vcm-only'
