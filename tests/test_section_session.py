import json
import pytest

from protocols.section_session import SectionContractError, SectionSessionMetaV1


def citation():
    return {"collection_id": "c1", "document_id": "d1", "book_title": "Book", "page": 3, "section": "S"}


def metadata():
    return {
        "schema_name": "univai.section-session-meta", "schema_version": "1.0.0",
        "learner_id": "u1", "programme_id": "p1", "programme_title": "Programme", "course_id": "c1", "week": 2,
        "lecture_id": "lecture-2", "plan_version": 4, "lecture_completed": True,
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


def test_section_requires_completed_lecture_and_owner():
    bad = metadata(); bad["lecture_completed"] = False
    with pytest.raises(SectionContractError, match="completed"):
        SectionSessionMetaV1.from_room_metadata(json.dumps(bad), authenticated_learner_id="u1")
    with pytest.raises(SectionContractError, match="own"):
        SectionSessionMetaV1.from_room_metadata(json.dumps(metadata()), authenticated_learner_id="u2")
