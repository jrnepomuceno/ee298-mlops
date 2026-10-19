#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UNIT_NAME="pi5-vcm-inference.service"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_PATH="$UNIT_DIR/$UNIT_NAME"
RUNTIME_ENV_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/pi5-vcm/environment"
DUMMY_TIME="${PI5_VCM_DUMMY_TIME:-0}"
DUMMY_INTENT="${PI5_VCM_DUMMY_INTENT:-}"

if [[ ! -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    printf 'Runner is not set up yet; run bash "%s/setup_pi.sh" first.\n' "$SCRIPT_DIR" >&2
    exit 1
fi
if ! systemctl --user is-system-running >/dev/null 2>&1; then
    printf 'systemd user manager is not running for %s.\n' "$USER" >&2
    exit 1
fi
if [[ "$DUMMY_TIME" != "0" && "$DUMMY_TIME" != "1" ]]; then
    printf 'PI5_VCM_DUMMY_TIME must be 0 or 1.\n' >&2
    exit 1
fi
if [[ "$DUMMY_TIME" == "1" && -n "$DUMMY_INTENT" ]]; then
    printf 'PI5_VCM_DUMMY_TIME and PI5_VCM_DUMMY_INTENT cannot both be enabled.\n' >&2
    exit 1
fi
if [[ -n "$DUMMY_INTENT" ]]; then
    case "$DUMMY_INTENT" in
        PLAY_MUSIC|WEATHER|TIME|LIGHT_ON|LIGHT_OFF|PAUSE|STOP|NEXT|VOLUME_UP|VOLUME_DOWN|CALL|MESSAGE|LIST_REMINDERS|TIMER|ALARM|TEMPERATURE|BRIGHTNESS|COLOR|CREATE_REMINDER|OUT_OF_SCOPE)
            ;;
        *)
            printf 'PI5_VCM_DUMMY_INTENT must be a supported uppercase intent name.\n' >&2
            exit 1
            ;;
    esac
fi

RUNNER_ARGS=""
if [[ "$DUMMY_TIME" == "1" ]]; then
    RUNNER_ARGS=" --dummy-time"
elif [[ -n "$DUMMY_INTENT" ]]; then
    RUNNER_ARGS=" --dummy-intent $DUMMY_INTENT"
fi

mkdir -p "$UNIT_DIR"
cat > "$UNIT_PATH" <<EOF
[Unit]
Description=Pi 5 Alexa wake-word acknowledgement listener
After=pipewire.service wireplumber.service
Wants=pipewire.service wireplumber.service

[Service]
Type=simple
WorkingDirectory=$SCRIPT_DIR
EnvironmentFile=-$RUNTIME_ENV_FILE
ExecStart=/usr/bin/bash $SCRIPT_DIR/run_pi.sh$RUNNER_ARGS
Restart=on-failure
RestartSec=5
KillSignal=SIGINT
TimeoutStopSec=15
StandardOutput=journal
StandardError=journal
SyslogIdentifier=pi5-vcm-inference

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable "$UNIT_NAME"
systemctl --user restart "$UNIT_NAME"
systemctl --user --no-pager --full status "$UNIT_NAME"

linger="$(loginctl show-user "$USER" -p Linger --value 2>/dev/null || true)"
if [[ "$linger" != "yes" ]]; then
    printf '\nThe listener is running in the current user session.\n'
    printf 'For it to survive logout and start at boot, enable lingering once:\n'
    printf '  sudo loginctl enable-linger %s\n' "$USER"
fi

printf '\nLive logs:\n  journalctl --user -u %s -f\n' "$UNIT_NAME"
if [[ "$DUMMY_TIME" == "1" ]]; then
    printf '\nWARNING: persistent dummy TIME mode is enabled for every voiced command.\n'
elif [[ -n "$DUMMY_INTENT" ]]; then
    printf '\nWARNING: persistent dummy %s mode is enabled for every voiced command.\n' "$DUMMY_INTENT"
else
    printf '\nNormal standby mode is enabled; dummy TIME mode is off.\n'
fi
