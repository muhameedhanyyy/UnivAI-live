# Live worker deployment

Build with `docker build -t univai-live:final .`. The image is repository-local and starts the standalone simulator by default. An integrated deployment must override the command and supply LiveKit, Agent, STT and TTS configuration. `python health.py` reports process liveness separately from dependency readiness and exits non-zero until all dependencies are configured.
