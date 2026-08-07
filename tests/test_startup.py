import pytest

import startup
from startup import ArtifactIndex, StartupStage, StartupTrace


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
