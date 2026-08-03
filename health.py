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
        _model("stt", os.getenv("STT_MODEL_PATH"), allow_named=os.getenv("STT_MODEL_SIZE")),
        _model("tts", os.getenv("KOKORO_MODEL") or os.getenv("PIPER_MODEL")),
        _endpoint("agent", os.getenv("RAG_URL") or os.getenv("AGENT_URL")),
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
    return DependencyHealth(name, True, "configured")


def _model(name: str, path: str | None, *, allow_named: str | None = None) -> DependencyHealth:
    if path:
        return DependencyHealth(name, Path(path).expanduser().is_file(), "available" if Path(path).expanduser().is_file() else "model_missing")
    if allow_named:
        return DependencyHealth(name, True, "named_model_configured")
    return DependencyHealth(name, False, "not_configured")


if __name__ == "__main__":
    payload = health_payload()
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["ready"] else 1)
