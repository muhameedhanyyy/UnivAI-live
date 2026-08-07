FROM ghcr.io/astral-sh/uv:0.10.10 AS uv
FROM python:3.12-slim
WORKDIR /app
COPY --from=uv /uv /uvx /bin/

# Production image: build from the UnivAI campus root with
#   docker build -f UnivAI-live/Dockerfile .
COPY UnivAI-live/pyproject.toml UnivAI-live/uv.lock ./
RUN uv sync --frozen --no-dev --extra integrated
COPY services /opt/univai/services
COPY UnivAI-live .
RUN adduser --disabled-password --gecos "" --uid 10001 liveworker \
    && mkdir -p /var/cache/univai-live/audio \
    && chown -R liveworker:liveworker /var/cache/univai-live

ENV PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH \
    UNIVAI_MODE=integrated \
    UNIVAI_ENV=production \
    UNIVAI_INTEGRATION_ROOT=/opt/univai \
    LIVE_AUDIO_CACHE_DIR=/var/cache/univai-live/audio
EXPOSE 8081
HEALTHCHECK --interval=15s --timeout=8s --start-period=20s --retries=4 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8081/', timeout=2); from health import health_payload; assert health_payload()['ready']"]
USER liveworker
CMD ["python", "worker.py", "start"]
