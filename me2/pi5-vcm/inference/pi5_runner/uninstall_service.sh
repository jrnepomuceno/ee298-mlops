#!/usr/bin/env bash
set -Eeuo pipefail

UNIT_NAME="pi5-vcm-inference.service"
UNIT_PATH="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/$UNIT_NAME"

systemctl --user disable --now "$UNIT_NAME" 2>/dev/null || true
rm -f -- "$UNIT_PATH"
systemctl --user daemon-reload
systemctl --user reset-failed "$UNIT_NAME" 2>/dev/null || true

printf 'Stopped and removed the user service.\n'
printf 'The runner folder and downloaded model assets were left intact.\n'
printf 'To remove those too, delete only the pi5_runner folder.\n'
