#!/usr/bin/env bash
set -eu

if [ -n "${MODEL_CHECKPOINT_URL:-}" ] && [ ! -f "${MODEL_CHECKPOINT:-/var/data/generator.pt}" ]; then
    checkpoint="${MODEL_CHECKPOINT:-/var/data/generator.pt}"
    mkdir -p "$(dirname "$checkpoint")"
    curl --fail --location --silent --show-error \
        --output "${checkpoint}.download" \
        "$MODEL_CHECKPOINT_URL"
    mv "${checkpoint}.download" "$checkpoint"
fi

exec python -m uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --proxy-headers
