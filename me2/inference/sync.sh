#!/usr/bin/env bash
# Re-vendor the canonical project modules into this self-contained folder.
#
# config.py / model/ / utils/ here are COPIES of the project-root modules so
# this folder can be dropped onto the Pi5 alongside best.pt and run standalone.
# Run this after changing config.py, model/*.py, or utils/*.py at the root.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

cp "${ROOT}/config.py" "${HERE}/config.py"
rm -rf "${HERE}/model"
cp -R "${ROOT}/model" "${HERE}/model"
mkdir -p "${HERE}/utils"
cp "${ROOT}/utils/"*.py "${HERE}/utils/"

echo "synced config.py, model/, utils/ from ${ROOT}"
