from pathlib import Path

from health import health_payload


class _Connection:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_health_distinguishes_liveness_and_readiness(monkeypatch):
    for name in ("LIVEKIT_URL", "DATABASE_URL", "STT_MODEL_PATH", "STT_MODEL_SIZE", "KOKORO_MODEL", "PIPER_MODEL", "RAG_MCP_URL", "RAG_URL", "AGENT_URL"):
        monkeypatch.delenv(name, raising=False)
    payload = health_payload()
    assert payload["live"] is True
    assert payload["ready"] is False
    assert {item["name"] for item in payload["dependencies"]} == {"livekit", "database", "stt", "tts", "agent"}


def test_readiness_checks_reachability_and_local_models(monkeypatch, tmp_path: Path):
    stt = tmp_path / "stt.bin"; stt.write_bytes(b"model")
    tts = tmp_path / "tts.onnx"; tts.write_bytes(b"model")
    monkeypatch.setenv("LIVEKIT_URL", "ws://livekit:7880")
    monkeypatch.setenv("DATABASE_URL", "postgresql://db:5432/univai")
    monkeypatch.setenv("RAG_MCP_URL", "http://agent:8001")
    monkeypatch.setenv("STT_MODEL_PATH", str(stt))
    monkeypatch.setenv("KOKORO_MODEL", str(tts))
    monkeypatch.setattr("health.socket.create_connection", lambda *_args, **_kwargs: _Connection())
    assert health_payload()["ready"] is True
