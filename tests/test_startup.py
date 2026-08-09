import asyncio
import time

import pytest

import startup
from startup import ArtifactIndex, LazyDependencies, StartupStage, StartupTrace


def test_trace_requires_monotonic_stage_order():
    values = iter((0, 0, .1, .2))
    trace = StartupTrace(clock=lambda: next(values), trace_id="trace")
    trace.mark(StartupStage.ROOM_CONNECTED)
    trace.mark(StartupStage.METADATA_VALID)
    assert [mark.offset_ms for mark in trace.marks] == [0, 100, 200]
    with pytest.raises(ValueError, match="out of order"):
        trace.mark(StartupStage.TRACK_PUBLISHED)


def test_artifact_index_fails_closed(monkeypatch):
    monkeypatch.setattr(startup, "_db_fetch_all", lambda _sql: [])
    monkeypatch.setattr(startup, "_db_fetch_one", lambda _sql, _params: None)
    with pytest.raises(FileNotFoundError):
        ArtifactIndex().require("u1", 1)

    monkeypatch.setattr(startup, "_db_fetch_one", lambda _sql, _params: {"exists": 1})
    assert ArtifactIndex().require("u1", 1) == ("u1", 1)


def test_concurrent_stt_waiters_share_one_model_load(monkeypatch):
    calls = 0
    model = object()

    def load():
        nonlocal calls
        calls += 1
        time.sleep(.03)
        return model

    monkeypatch.setattr(LazyDependencies, "_load_stt", staticmethod(load))

    async def scenario():
        dependencies = LazyDependencies()
        left, right = await asyncio.gather(dependencies.stt(), dependencies.stt())
        assert left is model and right is model
        assert await dependencies.stt() is model

    asyncio.run(scenario())
    assert calls == 1
