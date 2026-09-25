#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/../pi5venv/bin/python}"
REMOTE_HOST="${REMOTE_HOST:-daniel-pc.tailfa8657.ts.net}"
REMOTE_USER="${REMOTE_USER:-jdrne}"
REMOTE_PLATFORM="${REMOTE_PLATFORM:-windows}"
REMOTE_PYTHON="${REMOTE_PYTHON:-python}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_ed25519_jdrne_daniel_pc}"
REMOTE_ROOT="${REMOTE_ROOT:-TrainingGround}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
VERSION="${VERSION:-v${RUN_ID}}"
PI_HOST="${PI_HOST:-192.168.68.52}"
PI_USER="${PI_USER:-jdrnepomuceno9}"
PI_SSH_KEY="${PI_SSH_KEY:-}"
PI_ROOT="${PI_ROOT:-~/MyProjects/pi5-vcm}"
KNOWN_HOSTS_FILE="${KNOWN_HOSTS_FILE:-${HOME}/.ssh/pi5vcm_known_hosts}"
DIST_DIR="${ROOT_DIR}/dist"
BUNDLE="${DIST_DIR}/pi5-vcm-${VERSION}.tar.gz"
REMOTE_RUN="${REMOTE_ROOT}/${VERSION}"
PI_RUN="${PI_RUN:-${PI_ROOT}}"

SSH_OPTS=(-i "${SSH_KEY}" -o IdentitiesOnly=yes -o BatchMode=yes
    -o StrictHostKeyChecking=yes -o UserKnownHostsFile="${KNOWN_HOSTS_FILE}"
    -o ConnectTimeout=10)
SCP_OPTS=("${SSH_OPTS[@]}")
PI_SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=yes
    -o UserKnownHostsFile="${KNOWN_HOSTS_FILE}" -o ConnectTimeout=10)
PI_SCP_OPTS=("${PI_SSH_OPTS[@]}")
if [[ -n "${PI_SSH_KEY}" ]]; then
    PI_SSH_OPTS+=(-i "${PI_SSH_KEY}" -o IdentitiesOnly=yes)
    PI_SCP_OPTS=("${PI_SSH_OPTS[@]}")
fi

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || die "required file not found: $1"
}

validate_version() {
    [[ "${VERSION}" =~ ^[A-Za-z0-9._-]+$ ]] || \
        die "VERSION must contain only letters, numbers, '.', '_' or '-'"
}

validate_inputs() {
    validate_version
    [[ "${REMOTE_ROOT}" =~ ^[A-Za-z0-9._-]+$ ]] || \
        die "REMOTE_ROOT contains unsafe characters"
    [[ "${PI_ROOT}" =~ ^[A-Za-z0-9._/~:-]+$ ]] || \
        die "PI_ROOT contains unsafe characters"
    [[ "${REMOTE_PLATFORM}" == windows || "${REMOTE_PLATFORM}" == unix ]] || \
        die "REMOTE_PLATFORM must be windows or unix"
    local epochs="${EPOCHS:-10}" batch_size="${BATCH_SIZE:-32}"
    [[ "${epochs}" =~ ^[1-9][0-9]{0,2}$ ]] &&
        (( epochs <= 1000 )) || die "EPOCHS must be between 1 and 1000"
    [[ "${batch_size}" =~ ^[1-9][0-9]{0,2}$ ]] &&
        (( batch_size <= 256 )) || die "BATCH_SIZE must be between 1 and 256"
    [[ "${NUM_WORKERS:-0}" =~ ^[0-9]+$ ]] ||
        die "NUM_WORKERS must be a non-negative integer"
    case "${DEVICE:-auto}" in auto|cpu|cuda|mps) ;; *)
        die "DEVICE must be auto, cpu, cuda, or mps" ;;
    esac
    if [[ -n "${MANIFEST:-}" ]]; then
        [[ "${MANIFEST}" != /* && "${MANIFEST}" != *'..'* ]] || \
            die "MANIFEST must be a relative path without '..'"
        [[ "${MANIFEST}" =~ ^[A-Za-z0-9._/-]+\.(jsonl|csv)$ ]] || \
            die "MANIFEST must be a relative .jsonl or .csv path"
    fi
    [[ -f "${KNOWN_HOSTS_FILE}" ]] || \
        die "known-hosts file not found: ${KNOWN_HOSTS_FILE}"
}

validate_manifest() {
    if [[ -n "${MANIFEST:-}" ]]; then
        [[ "${MANIFEST}" != /* && "${MANIFEST}" != *'..'* ]] || \
            die "MANIFEST must be a relative path without '..'"
        [[ "${MANIFEST}" =~ ^[A-Za-z0-9._/-]+\.(jsonl|csv)$ ]] || \
            die "MANIFEST must be a relative .jsonl or .csv path"
    fi
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1"
    else
        shasum -a 256 "$1"
    fi
}

compile() {
    if [[ "${PYTHON_BIN}" == */* ]]; then
        require_file "${PYTHON_BIN}"
    else
        command -v "${PYTHON_BIN}" >/dev/null 2>&1 ||
            die "Python command not found: ${PYTHON_BIN}"
    fi
    "${PYTHON_BIN}" -m compileall -q \
        "${ROOT_DIR}/config.py" "${ROOT_DIR}/dataset.py" \
        "${ROOT_DIR}/dataloader.py" "${ROOT_DIR}/model.py" \
        "${ROOT_DIR}/train.py" "${ROOT_DIR}/main.py" "${ROOT_DIR}/utils"
    printf '[compile] python sources are valid\n'
}

package() {
    validate_version
    validate_manifest
    compile
    mkdir -p "${DIST_DIR}"
    local staging
    staging="$(mktemp -d)"
    mkdir -p "${staging}/pi5-vcm-${VERSION}/utils"
    cp "${ROOT_DIR}"/{config.py,dataset.py,dataloader.py,model.py,train.py,main.py} \
        "${staging}/pi5-vcm-${VERSION}/"
    cp "${ROOT_DIR}"/utils/*.py "${staging}/pi5-vcm-${VERSION}/utils/"
    cp "${ROOT_DIR}/requirements-runtime.txt" "${staging}/pi5-vcm-${VERSION}/"
    if [[ -n "${MANIFEST:-}" ]]; then
        require_file "${MANIFEST}"
        mkdir -p "${staging}/pi5-vcm-${VERSION}/data"
        cp "${MANIFEST}" "${staging}/pi5-vcm-${VERSION}/data/$(basename "${MANIFEST}")"
    fi
    tar -czf "${BUNDLE}" -C "${staging}" "pi5-vcm-${VERSION}"
    rm -rf "${staging}"
    printf '[package] %s\n' "${BUNDLE}"
}

deploy() {
    validate_inputs
    [[ -f "${BUNDLE}" ]] || package
    local mkdir_command extract_command
    case "${REMOTE_PLATFORM}" in
        windows)
            mkdir_command="if not exist \"${REMOTE_RUN}\" mkdir \"${REMOTE_RUN}\""
            extract_command="tar -xzf \"${REMOTE_RUN}/bundle.tar.gz\" -C \"${REMOTE_RUN}\" --strip-components=1"
            ;;
        unix)
            mkdir_command="mkdir -p '${REMOTE_RUN}'"
            extract_command="tar -xzf '${REMOTE_RUN}/bundle.tar.gz' -C '${REMOTE_RUN}' --strip-components=1"
            ;;
        *)
            die "REMOTE_PLATFORM must be windows or unix"
            ;;
    esac
    ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "${mkdir_command}"
    scp "${SCP_OPTS[@]}" "${BUNDLE}" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_RUN}/bundle.tar.gz"
    ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "${extract_command}"
    printf '[deploy] %s:%s\n' "${REMOTE_HOST}" "${REMOTE_RUN}"
}

train_remote() {
    validate_inputs
    deploy
    local train_args=(--epochs "${EPOCHS:-10}" --batch-size "${BATCH_SIZE:-32}" \
        --num-workers "${NUM_WORKERS:-0}" --device "${DEVICE:-auto}")
    if [[ -n "${MANIFEST:-}" ]]; then
        train_args+=(--manifest "data/$(basename "${MANIFEST}")")
    fi
    local train_command
    if [[ "${REMOTE_PLATFORM}" == "windows" ]]; then
        train_command="cd /d \"${REMOTE_RUN}\" && ${REMOTE_PYTHON} main.py train ${train_args[*]}"
    else
        train_command="cd '${REMOTE_RUN}' && '${REMOTE_PYTHON}' main.py train ${train_args[*]}"
    fi
    ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "${train_command}"
    printf '[train] completed on %s\n' "${REMOTE_HOST}"
}

fetch() {
    validate_inputs
    mkdir -p "${ROOT_DIR}/artifacts/${VERSION}"
    scp "${SCP_OPTS[@]}" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_RUN}/checkpoints/pi5-vcm-best.pt" \
        "${ROOT_DIR}/artifacts/${VERSION}/pi5-vcm-best.pt"
    scp "${SCP_OPTS[@]}" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_RUN}/checkpoints/pi5-vcm-history.json" \
        "${ROOT_DIR}/artifacts/${VERSION}/pi5-vcm-history.json"
    cp "${ROOT_DIR}/requirements-runtime.txt" "${ROOT_DIR}/artifacts/${VERSION}/"
    sha256_file "${ROOT_DIR}/artifacts/${VERSION}/pi5-vcm-best.pt" \
        > "${ROOT_DIR}/artifacts/${VERSION}/pi5-vcm-best.pt.sha256"
    sha256_file "${ROOT_DIR}/artifacts/${VERSION}/pi5-vcm-history.json" \
        > "${ROOT_DIR}/artifacts/${VERSION}/pi5-vcm-history.json.sha256"
    printf '[fetch] artifacts/%s\n' "${VERSION}"
}

deploy_pi() {
    validate_inputs
    local artifact_dir="${ROOT_DIR}/artifacts/${VERSION}"
    [[ -f "${artifact_dir}/pi5-vcm-best.pt" ]] || fetch
    local backup_tag
    backup_tag="$(date -u +%Y-%m-%d)"

    ssh "${PI_SSH_OPTS[@]}" "${PI_USER}@${PI_HOST}" \
        "mkdir -p ${PI_RUN}/utils ${PI_RUN}/inference"
    ssh "${PI_SSH_OPTS[@]}" "${PI_USER}@${PI_HOST}" \
        "set -e; if [ -f ${PI_RUN}/inference/best.pt ]; then cp ${PI_RUN}/inference/best.pt ${PI_RUN}/inference/${backup_tag}_best.pt.old; fi"
    scp "${PI_SCP_OPTS[@]}" \
        "${artifact_dir}/pi5-vcm-best.pt" \
        "${PI_USER}@${PI_HOST}:${PI_RUN}/inference/best.pt"
    scp "${PI_SCP_OPTS[@]}" \
        "${artifact_dir}/pi5-vcm-history.json" \
        "${artifact_dir}/pi5-vcm-best.pt.sha256" "${artifact_dir}/pi5-vcm-history.json.sha256" \
        "${ROOT_DIR}/requirements-runtime.txt" \
        "${PI_USER}@${PI_HOST}:${PI_RUN}/"
    scp "${PI_SCP_OPTS[@]}" \
        "${ROOT_DIR}"/{config.py,dataset.py,dataloader.py,model.py,train.py,main.py} \
        "${PI_USER}@${PI_HOST}:${PI_RUN}/"
    scp "${PI_SCP_OPTS[@]}" "${ROOT_DIR}"/utils/*.py \
        "${PI_USER}@${PI_HOST}:${PI_RUN}/utils/"
    ssh "${PI_SSH_OPTS[@]}" "${PI_USER}@${PI_HOST}" \
        "cd '${PI_RUN}' && sha256sum -c pi5-vcm-best.pt.sha256 && sha256sum -c pi5-vcm-history.json.sha256"
    printf '[deploy-pi] %s:%s\n' "${PI_HOST}" "${PI_RUN}"
}

usage() {
    printf '%s\n' \
    'usage: ./pipeline.sh {compile|package|deploy|train-remote|fetch|deploy-pi|all|train-deploy-pi}' \
        '  MANIFEST=... EPOCHS=... REMOTE_PYTHON=... ./pipeline.sh train-remote' \
    '  VERSION=v1.0.0 PI_USER=pi PI_SSH_KEY=... ./pipeline.sh deploy-pi' \
    '  VERSION=v1.0.0 EPOCHS=30 ./pipeline.sh train-deploy-pi'
}

command="${1:-}"
case "${command}" in
    compile) compile ;;
    package) package ;;
    deploy) deploy ;;
    train-remote) train_remote ;;
    fetch) fetch ;;
    deploy-pi) deploy_pi ;;
    all) train_remote && fetch ;;
    train-deploy-pi) train_remote && fetch && deploy_pi ;;
    *) usage; exit 2 ;;
esac