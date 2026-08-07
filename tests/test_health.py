from pathlib import Path

import pytest

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


# Readiness reports on the engine that will actually speak, so the test has to
# name it. Left unset, the answer came from whatever the developer's .env said:
# importing tts.py calls load_dotenv() at module scope, so a suite that touched
# TTS at all put TTS_LIVE_ENGINE=piper into the environment and this test then
# looked for a Kokoro model the worker was never going to load. It passed alone,
# passed in CI (no .env), and failed for anyone running the full suite locally.
@pytest.mark.parametrize(
    "engine, model_variable",
    [("kokoro", "KOKORO_MODEL"), ("piper", "PIPER_MODEL")],
)
def test_readiness_checks_reachability_and_local_models(
    monkeypatch, tmp_path: Path, engine: str, model_variable: str
):
    stt = tmp_path / "stt.bin"; stt.write_bytes(b"model")
    tts = tmp_path / "tts.onnx"; tts.write_bytes(b"model")
    monkeypatch.setenv("LIVEKIT_URL", "ws://livekit:7880")
    monkeypatch.setenv("DATABASE_URL", "postgresql://db:5432/univai")
    monkeypatch.setenv("RAG_MCP_URL", "http://agent:8001")
    monkeypatch.setenv("STT_MODEL_PATH", str(stt))
    monkeypatch.setenv("TTS_LIVE_ENGINE", engine)
    monkeypatch.setenv(model_variable, str(tts))
    monkeypatch.setattr("health.socket.create_connection", lambda *_args, **_kwargs: _Connection())
    assert health_payload()["ready"] is True


def test_readiness_follows_the_live_engine_not_the_other_ones_model(monkeypatch, tmp_path: Path):
    """A present Kokoro model must not report ready while Piper is the voice."""
    stt = tmp_path / "stt.bin"; stt.write_bytes(b"model")
    kokoro = tmp_path / "kokoro.onnx"; kokoro.write_bytes(b"model")
    monkeypatch.setenv("LIVEKIT_URL", "ws://livekit:7880")
    monkeypatch.setenv("DATABASE_URL", "postgresql://db:5432/univai")
    monkeypatch.setenv("RAG_MCP_URL", "http://agent:8001")
    monkeypatch.setenv("STT_MODEL_PATH", str(stt))
    monkeypatch.setenv("TTS_LIVE_ENGINE", "piper")
    monkeypatch.setenv("KOKORO_MODEL", str(kokoro))
    monkeypatch.setenv("PIPER_MODEL", str(tmp_path / "absent.onnx"))
    monkeypatch.setattr("health.socket.create_connection", lambda *_args, **_kwargs: _Connection())

    payload = health_payload()

    assert payload["ready"] is False
    voice = next(item for item in payload["dependencies"] if item["name"] == "tts")
    assert voice["reason"] == "model_missing"
