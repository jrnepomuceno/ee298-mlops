#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UNIT_NAME="pi5-vcm-inference.service"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_PATH="$UNIT_DIR/$UNIT_NAME"

if [[ ! -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    printf 'Runner is not set up yet; run bash "%s/setup_pi.sh" first.\n' "$SCRIPT_DIR" >&2
    exit 1
fi
if ! systemctl --user is-system-running >/dev/null 2>&1; then
    printf 'systemd user manager is not running for %s.\n' "$USER" >&2
    exit 1
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
ExecStart=/usr/bin/bash $SCRIPT_DIR/run_pi.sh
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
systemctl --user enable --now "$UNIT_NAME"
systemctl --user --no-pager --full status "$UNIT_NAME"

linger="$(loginctl show-user "$USER" -p Linger --value 2>/dev/null || true)"
if [[ "$linger" != "yes" ]]; then
    printf '\nThe listener is running in the current user session.\n'
    printf 'For it to survive logout and start at boot, enable lingering once:\n'
    printf '  sudo loginctl enable-linger %s\n' "$USER"
fi

printf '\nLive logs:\n  journalctl --user -u %s -f\n' "$UNIT_NAME"
