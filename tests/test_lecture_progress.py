from datetime import datetime, timezone
from types import ModuleType
import sys

import pytest

from lecture_progress import (
    LectureAdmissionClosed,
    LectureCheckpoint,
    LectureProgressRepository,
    replay_start,
)


def test_resume_replays_exactly_three_previous_sentences():
    assert replay_start(12, 30) == 9
    assert LectureCheckpoint(12, 30).replay_from == 9


def test_early_resume_never_underflows():
    assert replay_start(2, 30) == 0
    assert replay_start(0, 30) == 0


def test_corrupt_or_stale_checkpoint_is_clamped_to_script():
    assert replay_start(99, 10) == 7
    assert replay_start(-5, 10) == 0
    assert replay_start(8, -1) == 0


def test_replay_window_can_be_disabled_without_changing_checkpoint():
    assert replay_start(8, 10, replay_sentences=0) == 8


@pytest.mark.parametrize(
    ("row", "expected"),
    (({"first_admission": True}, True), ({"first_admission": False}, False)),
)
def test_real_presence_distinguishes_first_admission_from_rejoin(monkeypatch, row, expected):
    common = ModuleType("common")
    database = ModuleType("common.db")
    database.fetch_one = lambda _sql, _params: row
    monkeypatch.setitem(sys.modules, "common", common)
    monkeypatch.setitem(sys.modules, "common.db", database)

    repository = LectureProgressRepository(lecture_id=1, learner_id="learner")
    assert repository.ensure_joined(datetime.now(timezone.utc)) is expected


def test_actual_first_admission_fails_after_cutoff(monkeypatch):
    common = ModuleType("common")
    database = ModuleType("common.db")
    database.fetch_one = lambda _sql, _params: None
    monkeypatch.setitem(sys.modules, "common", common)
    monkeypatch.setitem(sys.modules, "common.db", database)

    repository = LectureProgressRepository(lecture_id=1, learner_id="learner")
    with pytest.raises(LectureAdmissionClosed):
        repository.ensure_joined(datetime.now(timezone.utc))


def test_first_admission_uses_the_confirmed_makeup_start_when_approved(monkeypatch):
    captured = {}
    common = ModuleType("common")
    database = ModuleType("common.db")

    def fetch_one(sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return {"first_admission": True}

    database.fetch_one = fetch_one
    monkeypatch.setitem(sys.modules, "common", common)
    monkeypatch.setitem(sys.modules, "common.db", database)

    joined_at = datetime.now(timezone.utc)
    repository = LectureProgressRepository(lecture_id=7, learner_id="learner")
    assert repository.ensure_joined(joined_at) is True

    assert "item.remedy = 'makeup_live'" in captured["sql"]
    assert "item.makeup_started_at IS NOT NULL" in captured["sql"]
    assert "COALESCE(makeup.makeup_started_at, l.starts_at)" in captured["sql"]
    assert captured["params"][:2] == (7, "learner")
