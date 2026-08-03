from pathlib import Path
import json
import pytest

from startup import ArtifactIndex, StartupStage, StartupTrace


def test_trace_requires_monotonic_stage_order():
    values = iter((0, 0, .1, .2))
    trace = StartupTrace(clock=lambda: next(values), trace_id="trace")
    trace.mark(StartupStage.ROOM_CONNECTED)
    trace.mark(StartupStage.METADATA_VALID)
    assert [mark.offset_ms for mark in trace.marks] == [0, 100, 200]
    with pytest.raises(ValueError, match="out of order"):
        trace.mark(StartupStage.TRACK_PUBLISHED)


def test_artifact_index_fails_closed(tmp_path: Path):
    folder = tmp_path / "u1" / "week-1"
    (folder / "audio").mkdir(parents=True)
    (folder / "script.json").write_text(json.dumps({"title": "L", "segments": []}))
    (folder / "audio" / "meta.json").write_text(json.dumps({"sample_rate": 24000}))
    with pytest.raises(FileNotFoundError): ArtifactIndex(tmp_path).require("u1", 1)
    (folder / "audio" / "s0-t0.npy").write_bytes(b"clip")
    assert ArtifactIndex(tmp_path).require("u1", 1) == folder
