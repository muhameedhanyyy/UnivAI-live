from health import health_payload


def test_health_distinguishes_liveness_and_readiness(monkeypatch):
    for name in ("LIVEKIT_URL", "STT_MODEL_PATH", "STT_MODEL_SIZE", "KOKORO_MODEL", "PIPER_MODEL", "RAG_URL", "AGENT_URL"):
        monkeypatch.delenv(name, raising=False)
    payload = health_payload()
    assert payload["live"] is True
    assert payload["ready"] is False
    assert {item["name"] for item in payload["dependencies"]} == {"livekit", "stt", "tts", "agent"}
