#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname -- "$SCRIPT_DIR")"
PACKAGE_DIR_NAME="$(basename -- "$SCRIPT_DIR")"
OUTPUT_DIR="${1:-$SCRIPT_DIR/dist}"

if [[ $# -gt 2 ]]; then
    printf 'Usage: bash package_pi.sh [output-directory] [version]\n' >&2
    exit 2
fi

if [[ $# -eq 2 ]]; then
    PACKAGE_VERSION="$2"
else
    PACKAGE_VERSION="$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || date -u +%Y%m%d%H%M%S)"
    if [[ -n "$(git -C "$SCRIPT_DIR" status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
        PACKAGE_VERSION="${PACKAGE_VERSION}-dirty"
    fi
fi

if [[ ! "$PACKAGE_VERSION" =~ ^[A-Za-z0-9._-]+$ ]]; then
    printf 'Package version may contain only letters, numbers, dots, underscores, and hyphens.\n' >&2
    exit 2
fi

command -v tar >/dev/null || {
    printf 'tar is required to build the deployment package.\n' >&2
    exit 1
}
if command -v shasum >/dev/null; then
    CHECKSUM_TOOL=shasum
elif command -v sha256sum >/dev/null; then
    CHECKSUM_TOOL=sha256sum
else
    printf 'shasum or sha256sum is required to create the package checksum.\n' >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd -- "$OUTPUT_DIR" && pwd)"
ARCHIVE="$OUTPUT_DIR/pi5-vcm-inference-${PACKAGE_VERSION}.tar.gz"
TEMP_ARCHIVE="$(mktemp "$OUTPUT_DIR/.pi5-vcm-inference.XXXXXX")"
trap 'rm -f -- "$TEMP_ARCHIVE"' EXIT

tar -czf "$TEMP_ARCHIVE" \
    --exclude="$PACKAGE_DIR_NAME/.venv" \
    --exclude="$PACKAGE_DIR_NAME/models" \
    --exclude="$PACKAGE_DIR_NAME/.config" \
    --exclude="$PACKAGE_DIR_NAME/dist" \
    --exclude="$PACKAGE_DIR_NAME/wireplumber" \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    -C "$PARENT_DIR" "$PACKAGE_DIR_NAME"

mv -- "$TEMP_ARCHIVE" "$ARCHIVE"
trap - EXIT
chmod 644 "$ARCHIVE"
(
    cd "$OUTPUT_DIR"
    if [[ "$CHECKSUM_TOOL" == shasum ]]; then
        "$CHECKSUM_TOOL" -a 256 "$(basename -- "$ARCHIVE")"
    else
        "$CHECKSUM_TOOL" "$(basename -- "$ARCHIVE")"
    fi > "$(basename -- "$ARCHIVE").sha256"
)

printf 'Created deployment package:\n  %s\n' "$ARCHIVE"
printf 'Created SHA-256 checksum:\n  %s.sha256\n' "$ARCHIVE"
