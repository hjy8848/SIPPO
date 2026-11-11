#!/usr/bin/env bash
set -euo pipefail

: "${WAIT_LAUNCH_AUTHORIZED:?Set WAIT_LAUNCH_AUTHORIZED=true after explicit authorization}"
: "${WAIT_GPU_INDEX:?Set the physical GPU index}"
: "${WAIT_GPU_UUID:?Set the expected physical GPU UUID}"
: "${WAIT_LAUNCHER:?Set the audited launcher path}"
: "${WAIT_RUN_MATCH:?Set a unique process match}"
: "${WAIT_ARTIFACT_PATH:?Set the primary output path}"

[[ "${WAIT_LAUNCH_AUTHORIZED}" == "true" ]] || {
    echo "[ERROR] Read-only queue launch is not explicitly authorized" >&2
    exit 1
}

interval="${WAIT_INTERVAL_SECONDS:-30}"
[[ "${interval}" =~ ^[1-9][0-9]*$ ]] || {
    echo "[ERROR] WAIT_INTERVAL_SECONDS must be a positive integer" >&2
    exit 1
}
[[ -f "${WAIT_LAUNCHER}" ]] || {
    echo "[ERROR] Missing launcher: ${WAIT_LAUNCHER}" >&2
    exit 1
}

timestamp() {
    date --iso-8601=seconds
}

actual_uuid="$(
    nvidia-smi -i "${WAIT_GPU_INDEX}" --query-gpu=uuid --format=csv,noheader |
        tr -d '[:space:]'
)"
[[ "${actual_uuid}" == "${WAIT_GPU_UUID}" ]] || {
    echo "[ERROR] GPU ${WAIT_GPU_INDEX} UUID ${actual_uuid} != ${WAIT_GPU_UUID}" >&2
    exit 1
}

echo "[$(timestamp)] [INFO] Waiting read-only for GPU ${WAIT_GPU_INDEX} (${WAIT_GPU_UUID})"
while nvidia-smi -i "${WAIT_GPU_INDEX}" \
    --query-compute-apps=pid --format=csv,noheader | grep -q .; do
    sleep "${interval}"
done

if pgrep -af "${WAIT_RUN_MATCH}" >/dev/null; then
    echo "[ERROR] Matching run already exists: ${WAIT_RUN_MATCH}" >&2
    exit 1
fi
[[ ! -e "${WAIT_ARTIFACT_PATH}" ]] || {
    echo "[ERROR] Refusing to overwrite existing artifact: ${WAIT_ARTIFACT_PATH}" >&2
    exit 1
}

actual_uuid="$(
    nvidia-smi -i "${WAIT_GPU_INDEX}" --query-gpu=uuid --format=csv,noheader |
        tr -d '[:space:]'
)"
[[ "${actual_uuid}" == "${WAIT_GPU_UUID}" ]]
if nvidia-smi -i "${WAIT_GPU_INDEX}" \
    --query-compute-apps=pid --format=csv,noheader | grep -q .; then
    echo "[$(timestamp)] [INFO] GPU became busy during handoff; resuming wait" >&2
    exec bash "$0"
fi

echo "[$(timestamp)] [INFO] GPU ${WAIT_GPU_INDEX} is empty; handing off to ${WAIT_LAUNCHER}"
exec bash "${WAIT_LAUNCHER}"
