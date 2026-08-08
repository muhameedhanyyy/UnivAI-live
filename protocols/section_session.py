"""Versioned App/Agent/Live contract for grounded post-lecture sections."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

SECTION_META_SCHEMA = "univai.section-session-meta"
SECTION_META_VERSION = "1.0.0"
SECTION_REFERENCE_VERSION = "2.0.0"
SECTION_PACK_SCHEMA = "univai.section.pack"
SECTION_PACK_VERSION = "1.0.0"


class SectionContractError(ValueError):
    def __init__(self, message: str, *, field: str = "") -> None:
        super().__init__(message)
        self.field = field


@dataclass(frozen=True)
class SectionSessionReferenceV2:
    learner_id: str
    section_pack_id: str
    nonce: str

    @classmethod
    def from_room_metadata(
        cls, raw: str, *, authenticated_learner_id: str
    ) -> "SectionSessionReferenceV2":
        try:
            data = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise SectionContractError("section room metadata must be valid JSON", field="roomMetadata") from exc
        if not isinstance(data, dict):
            raise SectionContractError("section room metadata must be an object", field="roomMetadata")
        if data.get("schema_name") != SECTION_META_SCHEMA or data.get("schema_version") != SECTION_REFERENCE_VERSION:
            raise SectionContractError("unsupported section reference schema", field="schema_version")
        learner_id = _text(data, "learner_id")
        if learner_id != authenticated_learner_id:
            raise SectionContractError("authenticated learner does not own this section", field="learner_id")
        return cls(
            learner_id=learner_id,
            section_pack_id=_text(data, "section_pack_id"),
            nonce=_text(data, "nonce"),
        )


@dataclass(frozen=True)
class SectionSessionMetaV1:
    learner_id: str
    programme_id: str
    programme_title: str
    course_id: str
    week: int
    lecture_id: str
    plan_version: int
    # The lecture's scheduled end has passed. Not "the learner sat through it":
    # completion is only recorded when the Lecturer reaches the last line with
    # the learner present, so requiring it meant leaving a lecture early
    # forfeited that week's section for good. A section opens on the clock,
    # exactly like the week's quiz, and the App gates on the same rule.
    lecture_ended: bool
    pack: dict[str, Any]
    schema_name: str = SECTION_META_SCHEMA
    schema_version: str = SECTION_META_VERSION

    @classmethod
    def from_storage(
        cls,
        reference: SectionSessionReferenceV2,
        *,
        programme_id: str,
        programme_title: str,
        course_id: str,
        week: int,
        lecture_id: str,
        plan_version: int,
        lecture_ended: bool,
        pack: object,
    ) -> "SectionSessionMetaV1":
        payload = {
            "schema_name": SECTION_META_SCHEMA,
            "schema_version": SECTION_META_VERSION,
            "learner_id": reference.learner_id,
            "programme_id": programme_id,
            "programme_title": programme_title,
            "course_id": course_id,
            "week": week,
            "lecture_id": lecture_id,
            "plan_version": plan_version,
            "lecture_ended": lecture_ended,
            "pack": pack,
        }
        return cls.from_room_metadata(
            json.dumps(payload),
            authenticated_learner_id=reference.learner_id,
        )

    @classmethod
    def from_room_metadata(cls, raw: str, *, authenticated_learner_id: str) -> "SectionSessionMetaV1":
        try:
            data = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise SectionContractError("section room metadata must be valid JSON", field="roomMetadata") from exc
        if not isinstance(data, dict):
            raise SectionContractError("section room metadata must be an object", field="roomMetadata")
        if data.get("schema_name") != SECTION_META_SCHEMA or data.get("schema_version") != SECTION_META_VERSION:
            raise SectionContractError("unsupported section metadata schema", field="schema_version")
        learner = _text(data, "learner_id")
        if learner != authenticated_learner_id:
            raise SectionContractError("authenticated learner does not own this section", field="learner_id")
        week, version = data.get("week"), data.get("plan_version")
        if isinstance(week, bool) or not isinstance(week, int) or week < 1:
            raise SectionContractError("week must be a positive integer", field="week")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise SectionContractError("plan_version must be a positive integer", field="plan_version")
        if data.get("lecture_ended") is not True:
            raise SectionContractError("the linked lecture has not ended yet", field="lecture_ended")
        pack = validate_section_pack(data.get("pack"))
        expected = {
            "user_id": learner,
            "course_id": _text(data, "course_id"),
            "week_number": week,
            "topic_id": _text(data, "lecture_id"),
        }
        for field, value in expected.items():
            if pack.get(field) != value:
                raise SectionContractError(f"pack {field} does not match the exact session", field=field)
        if str(pack.get("plan_version")) != str(version):
            raise SectionContractError("pack plan_version does not match the exact session", field="plan_version")
        programme_title = _text(data, "programme_title")
        if pack.get("programme_title") != programme_title:
            raise SectionContractError("pack programme_title does not match the exact session", field="programme_title")
        return cls(
            learner_id=learner,
            programme_id=_text(data, "programme_id"),
            programme_title=programme_title,
            course_id=expected["course_id"],
            week=week,
            lecture_id=expected["topic_id"],
            plan_version=version,
            lecture_ended=True,
            pack=pack,
        )


def validate_section_pack(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SectionContractError("SectionPackV1 is required", field="pack")
    if value.get("schema_name") != SECTION_PACK_SCHEMA or value.get("schema_version") != SECTION_PACK_VERSION or value.get("session_type") != "section":
        raise SectionContractError("unsupported SectionPackV1 identity", field="pack.schema_version")
    for field in ("user_id", "course_id", "topic_id", "title"):
        _text(value, field)
    activities = value.get("activities")
    if not isinstance(activities, list) or not activities:
        raise SectionContractError("pack activities must be non-empty", field="pack.activities")
    for collection in (activities, value.get("examples", []), value.get("todos", [])):
        if not isinstance(collection, list):
            raise SectionContractError("section content collections must be arrays", field="pack")
        for item in collection:
            if not isinstance(item, dict) or not _citations(item):
                raise SectionContractError("every section item requires provenance", field="pack.citations")
            for step in item.get("steps", []):
                if not isinstance(step, dict) or not _citations(step):
                    raise SectionContractError("every worked step requires provenance", field="pack.citations")
    return value


def _text(data: dict, field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SectionContractError(f"{field} must be a non-empty string", field=field)
    return value.strip()


def _citations(item: dict) -> list[dict]:
    citations = item.get("citations")
    if not isinstance(citations, list) or not citations:
        return []
    return [citation for citation in citations if isinstance(citation, dict) and citation.get("document_id") and citation.get("page")]
