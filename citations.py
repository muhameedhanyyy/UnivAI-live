"""Bridge between qa.answer_question() raw dicts and AttributedAnswer.

``enrich_citations()`` is the single point where:

1.  The raw ``{answer, pages, model_used}`` dict from ``qa.py`` is upgraded to
    a typed ``AttributedAnswer``.
2.  Session identity (programme_id, course_id, lecture_id, plan_version) is
    attached to each citation record.
3.  The ``validate_attributed_answer()`` contract is enforced — a result that
    has no citations and is not a refusal or TROUBLE fallback will raise here
    rather than silently reaching the browser or TTS.

This module has no runtime dependencies beyond the stdlib and the
``protocols`` package so that it can be tested cheaply without any LiveKit,
TTS, STT, LLM, or database imports.
"""

from __future__ import annotations

from protocols.source_attribution import (
    AttributedAnswer,
    CitationRecord,
    build_spoken_citation,
    validate_attributed_answer,
)

# Kept in sync with qa.NOT_COVERED (lower-cased sentinel)
_NOT_COVERED_SENTINEL = "that is not covered in your book"

# Kept in sync with qa.TROUBLE (prefix check)
_TROUBLE_PREFIX = "I had trouble looking that up"


def enrich_citations(
    qa_result: dict,
    *,
    programme_id: str = "",
    course_id: str = "",
    lecture_id: str = "",
    plan_version: int | None = None,
) -> AttributedAnswer:
    """Build a validated ``AttributedAnswer`` from a raw ``qa.answer_question`` dict.

    Parameters
    ----------
    qa_result:
        Dict returned by ``qa.answer_question()``.  Expected keys:
        ``answer`` (str), ``pages`` (list[int]), ``model_used`` (str).
        Unknown extra keys are ignored.
    programme_id, course_id, lecture_id, plan_version:
        Session identity from ``LectureSessionMeta.as_citation_scope()``.
        All default to empty string when the session metadata was not
        available (e.g. standalone simulator).

    Returns
    -------
    AttributedAnswer
        Typed, validated attributed answer.

    Raises
    ------
    ValueError
        If the result is internally inconsistent (e.g. grounded answer with no
        citations).  This is a programming error in the Q&A pipeline, not a
        user-visible condition.
    """
    answer: str = qa_result.get("answer", "")
    raw_pages: list = qa_result.get("pages", [])
    raw_citations: list = qa_result.get("citations", [])
    model_used: str = qa_result.get("model_used", "")

    refused = _NOT_COVERED_SENTINEL in answer.lower()
    is_trouble = answer.startswith(_TROUBLE_PREFIX)

    citations: list[CitationRecord] = []
    if not refused and not is_trouble:
        candidates = raw_citations or [
            {"source": "", "page": page} for page in raw_pages
        ]
        seen: set[tuple[str, int | None]] = set()
        for raw in candidates:
            if not isinstance(raw, dict):
                continue
            source = raw.get("source")
            page = raw.get("page")
            normalized_page = page if isinstance(page, int) and page > 0 else None
            if not isinstance(source, str):
                source = ""
            identity = (source.strip(), normalized_page)
            if identity not in seen:
                seen.add(identity)
                citations.append(
                    CitationRecord(
                        source=source.strip(),
                        page=normalized_page,
                        programme_id=programme_id,
                        course_id=course_id,
                        lecture_id=lecture_id,
                        plan_version=plan_version,
                        chunk_id=str(raw.get("chunk_id") or ""),
                    )
                )

    spoken_citation = build_spoken_citation(citations)

    attributed = AttributedAnswer(
        answer=answer,
        citations=citations,
        model_used=model_used,
        refused=refused,
        spoken_citation=spoken_citation,
    )

    validate_attributed_answer(attributed)
    return attributed
