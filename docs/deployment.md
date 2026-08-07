# Live worker deployment

Build the production image from the UnivAI campus root because the integrated
worker imports the shared `services` package:

```bash
docker build -f UnivAI-live/Dockerfile -t univai-live:final .
```

The production image runs `python worker.py start` as an unprivileged user.
`Dockerfile.simulator` remains the explicitly separate, repository-local
standalone image. The Agents server exposes its LiveKit-connected health endpoint
on port 8081; the image health check also verifies PostgreSQL, RAG, configured
model files, and their network endpoints through `health.py`.

Mount the TTS models read-only at `/models` and a writable disposable audio
cache at `/var/cache/univai-live/audio`. Lecture and section content itself is
loaded from PostgreSQL and is not baked into or persisted by the worker image.
