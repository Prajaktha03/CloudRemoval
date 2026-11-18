#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${RAILWAY_PUBLIC_DOMAIN:-}" && -z "${FRONTEND_ORIGINS:-}" ]]; then
    export FRONTEND_ORIGINS="https://${RAILWAY_PUBLIC_DOMAIN}"
fi

if [[ -n "${MODEL_CHECKPOINT_URL:-}" && ! -f "${MODEL_CHECKPOINT}" ]]; then
    mkdir -p "$(dirname "${MODEL_CHECKPOINT}")"
    curl --fail --location --silent --show-error \
        --output "${MODEL_CHECKPOINT}.download" \
        "${MODEL_CHECKPOINT_URL}"
    mv "${MODEL_CHECKPOINT}.download" "${MODEL_CHECKPOINT}"
fi

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
backend_pid=$!
npm --prefix frontend run start -- --hostname 0.0.0.0 --port "${PORT}" &
frontend_pid=$!

shutdown() {
    kill "${backend_pid}" "${frontend_pid}" 2>/dev/null || true
    wait "${backend_pid}" "${frontend_pid}" 2>/dev/null || true
}
trap shutdown EXIT INT TERM

wait -n "${backend_pid}" "${frontend_pid}"
