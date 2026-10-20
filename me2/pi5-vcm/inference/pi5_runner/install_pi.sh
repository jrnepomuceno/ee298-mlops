#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--setup-only" ) ]]; then
    printf 'Usage: bash install_pi.sh [--setup-only]\n' >&2
    exit 2
fi

if [[ ! -f "$SCRIPT_DIR/run_pi.sh" || ! -f "$SCRIPT_DIR/setup_pi.sh" ]]; then
    printf 'The inference package is incomplete: run_pi.sh or setup_pi.sh is missing.\n' >&2
    exit 1
fi

bash "$SCRIPT_DIR/setup_pi.sh"

if [[ "${1:-}" == "--setup-only" ]]; then
    printf '\nRuntime setup is complete. Start manually with:\n'
    printf '  bash "%s/run_pi.sh"\n' "$SCRIPT_DIR"
    printf 'Or install the user service with:\n'
    printf '  bash "%s/install_service.sh"\n' "$SCRIPT_DIR"
    exit 0
fi

bash "$SCRIPT_DIR/install_service.sh"
