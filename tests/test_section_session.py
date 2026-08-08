import json
import pytest

from protocols.section_session import (
    SectionContractError,
    SectionSessionMetaV1,
    SectionSessionReferenceV2,
)


def citation():
    return {"collection_id": "c1", "document_id": "d1", "book_title": "Book", "page": 3, "section": "S"}


def metadata():
    return {
        "schema_name": "univai.section-session-meta", "schema_version": "1.0.0",
        "learner_id": "u1", "programme_id": "p1", "programme_title": "Programme", "course_id": "c1", "week": 2,
        "lecture_id": "lecture-2", "plan_version": 4, "lecture_ended": True,
        "pack": {"schema_name": "univai.section.pack", "schema_version": "1.0.0", "session_type": "section",
                 "user_id": "u1", "programme_title": "Programme", "course_id": "c1", "week_number": 2, "topic_id": "lecture-2", "plan_version": "4", "title": "Practice",
                 "activities": [{"order": 1, "title": "Try", "description": "Solve", "citations": [citation()]}],
                 "examples": [{"order": 1, "prompt": "Example", "citations": [citation()], "steps": [{"step": "1", "explanation": "Do it", "citations": [citation()]}]}],
                 "todos": [{"order": 1, "text": "Review", "citations": [citation()]}]},
    }


def test_exact_identity_and_provenance_are_required():
    meta = SectionSessionMetaV1.from_room_metadata(json.dumps(metadata()), authenticated_learner_id="u1")
    assert meta.plan_version == 4
    bad = metadata(); bad["pack"]["plan_version"] = "3"
    with pytest.raises(SectionContractError, match="plan_version"):
        SectionSessionMetaV1.from_room_metadata(json.dumps(bad), authenticated_learner_id="u1")


def test_section_requires_an_ended_lecture_and_owner():
    bad = metadata(); bad["lecture_ended"] = False
    with pytest.raises(SectionContractError, match="has not ended"):
        SectionSessionMetaV1.from_room_metadata(json.dumps(bad), authenticated_learner_id="u1")
    with pytest.raises(SectionContractError, match="own"):
        SectionSessionMetaV1.from_room_metadata(json.dumps(metadata()), authenticated_learner_id="u2")


def test_v2_room_metadata_is_only_a_storage_reference():
    raw = json.dumps({
        "schema_name": "univai.section-session-meta",
        "schema_version": "2.0.0",
        "learner_id": "u1",
        "section_pack_id": "9e066a6e-448b-4c22-a502-04fdab76f250",
        "nonce": "94f9a361-9495-47e6-af06-66d5b22577fb",
    })
    reference = SectionSessionReferenceV2.from_room_metadata(raw, authenticated_learner_id="u1")
    assert reference.section_pack_id == "9e066a6e-448b-4c22-a502-04fdab76f250"
    assert "activities" not in raw
    assert "examples" not in raw

    stored = metadata()
    meta = SectionSessionMetaV1.from_storage(
        reference,
        programme_id=stored["programme_id"],
        programme_title=stored["programme_title"],
        course_id=stored["course_id"],
        week=stored["week"],
        lecture_id=stored["lecture_id"],
        plan_version=stored["plan_version"],
        lecture_ended=stored["lecture_ended"],
        pack=stored["pack"],
    )
    assert meta.pack["title"] == "Practice"
