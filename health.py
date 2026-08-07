"""Liveness and dependency-specific readiness for the Live worker."""

from __future__ import annotations

import json
import os
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class DependencyHealth:
    name: str
    ready: bool
    reason: str


def dependency_health() -> list[DependencyHealth]:
    return [
        _endpoint("livekit", os.getenv("LIVEKIT_URL")),
        _endpoint("database", os.getenv("DATABASE_URL")),
        _model("stt", os.getenv("STT_MODEL_PATH"), allow_named=os.getenv("STT_MODEL_SIZE")),
        _model("tts", _tts_model_path()),
        _endpoint("agent", os.getenv("RAG_MCP_URL") or os.getenv("RAG_URL") or os.getenv("AGENT_URL")),
    ]


def health_payload() -> dict:
    dependencies = dependency_health()
    return {
        "live": True,
        "ready": all(item.ready for item in dependencies),
        "dependencies": [asdict(item) for item in dependencies],
    }


def _endpoint(name: str, value: str | None) -> DependencyHealth:
    if not value:
        return DependencyHealth(name, False, "not_configured")
    parsed = urlparse(value if "://" in value else f"tcp://{value}")
    if not parsed.hostname:
        return DependencyHealth(name, False, "invalid_endpoint")
    port = parsed.port or (443 if parsed.scheme in {"https", "wss"} else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=1.5):
            return DependencyHealth(name, True, "reachable")
    except OSError as exc:
        return DependencyHealth(name, False, f"unreachable:{type(exc).__name__}")


def _model(name: str, path: str | None, *, allow_named: str | None = None) -> DependencyHealth:
    if path:
        return DependencyHealth(name, Path(path).expanduser().is_file(), "available" if Path(path).expanduser().is_file() else "model_missing")
    if allow_named:
        # Faster Whisper is deliberately lazy because it is not on the
        # lecture's first-audio path. A configured model name is ready for that
        # design; loading it during container health probes would defeat it.
        return DependencyHealth(name, True, "configured_lazy_model")
    return DependencyHealth(name, False, "not_configured")


def _tts_model_path() -> str | None:
    engine = os.getenv("TTS_LIVE_ENGINE") or os.getenv("TTS_ENGINE", "kokoro")
    if engine.lower() == "piper":
        return os.getenv("PIPER_MODEL", "models/piper/en_US-lessac-medium.onnx")
    return os.getenv("KOKORO_MODEL", "models/kokoro/kokoro-v1.0.onnx")


if __name__ == "__main__":
    payload = health_payload()
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["ready"] else 1)
