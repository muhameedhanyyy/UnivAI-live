"""Tests for protocols.lecture_session: roomMetadata parsing and validation.

All tests run without LiveKit, databases, or model dependencies.
"""

from __future__ import annotations

import json
import os
import unittest

os.environ["UNIVAI_MODE"] = "standalone"

from protocols.lecture_session import (
    LectureSessionMeta,
    SessionMetadataError,
    STANDALONE_ROOM_METADATA,
    _sid_from_room_name,
)


VALID_METADATA: dict = {
    "programme_id": "programme-cs-2026",
    "course_id": "course-ai-101",
    "plan_version": "v2",
    "week": 3,
    "lecture_id": "lecture-week-3",
}

VALID_ROOM_NAME = "lecture-S-2026-000042-week-3"


class TestSidFromRoomName(unittest.TestCase):
    def test_standard_convention(self) -> None:
        self.assertEqual("S-2026-000042", _sid_from_room_name(VALID_ROOM_NAME))

    def test_sid_containing_dashes(self) -> None:
        self.assertEqual(
            "S-2026-000099",
            _sid_from_room_name("lecture-S-2026-000099-week-2"),
        )

    def test_unexpected_format_returns_empty(self) -> None:
        self.assertEqual("", _sid_from_room_name("wrong-room"))
        self.assertEqual("", _sid_from_room_name(""))


class TestFromRoomMetadataIntegrated(unittest.TestCase):
    """These tests temporarily switch to integrated mode."""

    def _run_integrated(self, fn):
        original = os.environ.get("UNIVAI_MODE", "standalone")
        os.environ["UNIVAI_MODE"] = "integrated"
        try:
            return fn()
        finally:
            os.environ["UNIVAI_MODE"] = original

    def test_valid_metadata_parses_correctly(self) -> None:
        meta_json = json.dumps(VALID_METADATA)
        def run():
            return LectureSessionMeta.from_room_metadata(VALID_ROOM_NAME, meta_json, sid="S-2026-000042")
        result = self._run_integrated(run)
        self.assertEqual("programme-cs-2026", result.programme_id)
        self.assertEqual("course-ai-101", result.course_id)
        self.assertEqual("v2", result.plan_version)
        self.assertEqual(3, result.week)
        self.assertEqual("lecture-week-3", result.lecture_id)
        self.assertEqual("S-2026-000042", result.sid)

    def test_extra_keys_are_ignored(self) -> None:
        data = dict(VALID_METADATA, _extra="ignored", _schema_version="1")
        def run():
            return LectureSessionMeta.from_room_metadata(VALID_ROOM_NAME, json.dumps(data))
        result = self._run_integrated(run)
        self.assertEqual("lecture-week-3", result.lecture_id)

    def test_absent_metadata_raises_in_integrated(self) -> None:
        def run():
            LectureSessionMeta.from_room_metadata(VALID_ROOM_NAME, None)
        with self.assertRaises(SessionMetadataError) as ctx:
            self._run_integrated(run)
        self.assertEqual("roomMetadata", ctx.exception.field)

    def test_empty_string_metadata_raises_in_integrated(self) -> None:
        def run():
            LectureSessionMeta.from_room_metadata(VALID_ROOM_NAME, "")
        with self.assertRaises(SessionMetadataError):
            self._run_integrated(run)

    def test_invalid_json_raises(self) -> None:
        def run():
            LectureSessionMeta.from_room_metadata(VALID_ROOM_NAME, "{not-json}")
        with self.assertRaises(SessionMetadataError) as ctx:
            self._run_integrated(run)
        self.assertEqual("roomMetadata", ctx.exception.field)

    def test_metadata_is_not_object_raises(self) -> None:
        def run():
            LectureSessionMeta.from_room_metadata(VALID_ROOM_NAME, json.dumps([1, 2]))
        with self.assertRaises(SessionMetadataError):
            self._run_integrated(run)

    def test_missing_programme_id_raises(self) -> None:
        data = dict(VALID_METADATA)
        del data["programme_id"]
        def run():
            LectureSessionMeta.from_room_metadata(VALID_ROOM_NAME, json.dumps(data))
        with self.assertRaises(SessionMetadataError) as ctx:
            self._run_integrated(run)
        self.assertEqual("programme_id", ctx.exception.field)

    def test_empty_course_id_raises(self) -> None:
        data = dict(VALID_METADATA, course_id="")
        def run():
            LectureSessionMeta.from_room_metadata(VALID_ROOM_NAME, json.dumps(data))
        with self.assertRaises(SessionMetadataError) as ctx:
            self._run_integrated(run)
        self.assertEqual("course_id", ctx.exception.field)

    def test_empty_plan_version_raises(self) -> None:
        data = dict(VALID_METADATA, plan_version="   ")
        def run():
            LectureSessionMeta.from_room_metadata(VALID_ROOM_NAME, json.dumps(data))
        with self.assertRaises(SessionMetadataError) as ctx:
            self._run_integrated(run)
        self.assertEqual("plan_version", ctx.exception.field)

    def test_week_zero_raises(self) -> None:
        data = dict(VALID_METADATA, week=0)
        def run():
            LectureSessionMeta.from_room_metadata(VALID_ROOM_NAME, json.dumps(data))
        with self.assertRaises(SessionMetadataError) as ctx:
            self._run_integrated(run)
        self.assertEqual("week", ctx.exception.field)

    def test_week_string_raises(self) -> None:
        data = dict(VALID_METADATA, week="three")
        def run():
            LectureSessionMeta.from_room_metadata(VALID_ROOM_NAME, json.dumps(data))
        with self.assertRaises(SessionMetadataError):
            self._run_integrated(run)


class TestFromRoomMetadataStandalone(unittest.TestCase):
    """In standalone mode absent metadata falls back to standalone_fixture()."""

    def test_absent_metadata_falls_back_to_fixture_in_standalone(self) -> None:
        # UNIVAI_MODE is already standalone in this file.
        result = LectureSessionMeta.from_room_metadata(
            "lecture-S-2026-000042-week-1", None
        )
        # Fields must be clearly labelled as fixture values.
        self.assertTrue(result.programme_id.startswith("programme-demo"))
        self.assertEqual("v1", result.plan_version)

    def test_valid_metadata_still_validated_in_standalone(self) -> None:
        result = LectureSessionMeta.from_room_metadata(
            VALID_ROOM_NAME, json.dumps(VALID_METADATA)
        )
        self.assertEqual("programme-cs-2026", result.programme_id)


class TestStandaloneFixture(unittest.TestCase):
    def test_fixture_returns_deterministic_values(self) -> None:
        meta = LectureSessionMeta.standalone_fixture()
        self.assertEqual("programme-demo-001", meta.programme_id)
        self.assertEqual("course-demo-001", meta.course_id)
        self.assertEqual("v1", meta.plan_version)
        self.assertEqual(1, meta.week)
        self.assertEqual("lecture-week-1", meta.lecture_id)
        self.assertEqual("S-2026-000042", meta.sid)

    def test_fixture_respects_custom_sid(self) -> None:
        meta = LectureSessionMeta.standalone_fixture(sid="S-test-999")
        self.assertEqual("S-test-999", meta.sid)

    def test_fixture_blocked_in_integrated(self) -> None:
        original = os.environ.get("UNIVAI_MODE", "standalone")
        os.environ["UNIVAI_MODE"] = "integrated"
        try:
            with self.assertRaises(RuntimeError):
                LectureSessionMeta.standalone_fixture()
        finally:
            os.environ["UNIVAI_MODE"] = original


class TestAsCitationScope(unittest.TestCase):
    def test_returns_expected_keys(self) -> None:
        meta = LectureSessionMeta.standalone_fixture()
        scope = meta.as_citation_scope()
        self.assertIn("programme_id", scope)
        self.assertIn("course_id", scope)
        self.assertIn("lecture_id", scope)
        self.assertIn("plan_version", scope)
        self.assertNotIn("sid", scope)
        self.assertNotIn("week", scope)


class TestCanonicalFixtureJson(unittest.TestCase):
    """The STANDALONE_ROOM_METADATA constant must parse as a valid fixture."""

    def test_canonical_constant_is_valid_json(self) -> None:
        data = json.loads(STANDALONE_ROOM_METADATA)
        self.assertIsInstance(data, dict)

    def test_canonical_constant_produces_valid_meta(self) -> None:
        result = LectureSessionMeta.from_room_metadata(
            "lecture-S-2026-000042-week-1",
            STANDALONE_ROOM_METADATA,
        )
        self.assertEqual("programme-demo-001", result.programme_id)
        self.assertEqual("v1", result.plan_version)


if __name__ == "__main__":
    unittest.main()
