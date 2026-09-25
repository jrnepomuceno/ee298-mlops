#!/usr/bin/env bash
# On-device inference wrapper for the Raspberry Pi 5.
#
#   ./run_pi5.sh --checkpoint best.pt --input cmd.wav
#   ./run_pi5.sh --checkpoint best.pt --self-test --json
#
# Uses the inference venv, then the project venv, else the system python3.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-${HERE}/venv/bin/python}"
[[ -x "${PY}" ]] || PY="${HERE}/../pi5venv/bin/python"
[[ -x "${PY}" ]] || PY="python3"
exec "${PY}" "${HERE}/infer.py" "$@"
