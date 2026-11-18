FROM node:22-bookworm-slim

ENV NEXT_TELEMETRY_DISABLED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    BACKEND_API_URL=http://127.0.0.1:8000

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y curl libgomp1 python3 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple "torch>=2.2" \
    && mkdir -p /data \
    && chown node:node /data

COPY --chown=node:node frontend/package.json frontend/package-lock.json ./frontend/
WORKDIR /app/frontend
RUN npm ci

WORKDIR /app
COPY --chown=node:node . .
RUN pip install --no-cache-dir -r requirements-deploy.txt

WORKDIR /app/frontend
RUN npm run build

WORKDIR /app
ENV MODEL_CHECKPOINT=/data/generator.pt \
    DATASET_ROOT=/data/raw \
    EXPERIMENTS_DB=/data/experiments.sqlite3 \
    SESSION_COOKIE_SECURE=true \
    PORT=3000

USER node
EXPOSE 3000
CMD ["bash", "scripts/start_container.sh"]
