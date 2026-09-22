#!/usr/bin/env bash
# Run the replay-first Pi harness from any working directory.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
PY="${PYTHON_BIN:-${ROOT}/pi5venv/bin/python}"
if [[ ! -x "${PY}" && -x "${ROOT}/../pi5venv/bin/python" ]]; then
	PY="${ROOT}/../pi5venv/bin/python"
fi
[[ -x "${PY}" ]] || PY="python3"
cd "${ROOT}"
exec "${PY}" -m rpi5.run "$@"
