"""Source attribution schema for the Q&A call chain.

Every spoken answer that is grounded in the textbook must carry citation
metadata from RAG retrieval all the way to the browser and to the TTS
render, so that the visible citation and the spoken citation are always
consistent with each other.

Definitions
-----------
CitationRecord
    One verified citation derived from a RAG hit.  Page and identity fields
    come from RAG metadata, NOT from model output.

AttributedAnswer
    The complete answer payload: text, citations, refusal flag, spoken
    citation suffix, and model attribution.

Validation contract
-------------------
* If ``refused=False`` and ``answer`` is not the TROUBLE fallback string,
  ``citations`` must be non-empty.
* If ``refused=True``, ``citations`` must be empty.
* An ``AttributedAnswer`` that violates these rules raises ``ValueError``
  from ``validate_attributed_answer()``.

The validation is enforced inside ``citations.enrich_citations()`` so it runs
on every production code path.  Tests may construct ``AttributedAnswer``
directly to verify the negative cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Kept in sync with qa.TROUBLE so that validate_attributed_answer can
# distinguish a genuine refusal from a degraded-fallback answer without
# importing qa (which imports heavy services).
_TROUBLE_PREFIX = "I had trouble looking that up"


@dataclass
class CitationRecord:
    """One source location cited in an answer.

    Attributes
    ----------
    page:
        Physical page number from the RAG hit metadata.
    programme_id, course_id, lecture_id, plan_version:
        Session identity fields forwarded from ``LectureSessionMeta``.
        Empty strings when the session metadata was unavailable (e.g. the
        standalone simulator).
    chunk_id:
        RAG chunk identifier when available; empty string otherwise.
    """

    page: int
    programme_id: str = ""
    course_id: str = ""
    lecture_id: str = ""
    plan_version: str = ""
    chunk_id: str = ""

    def as_dict(self) -> dict:
        """Serialise to a JSON-safe dict for the answer payload."""
        return {
            "page": self.page,
            "programme_id": self.programme_id,
            "course_id": self.course_id,
            "lecture_id": self.lecture_id,
            "plan_version": self.plan_version,
            "chunk_id": self.chunk_id,
        }


@dataclass
class AttributedAnswer:
    """Complete attributed answer for one Q&A turn.

    Attributes
    ----------
    answer:
        Spoken answer text — may already contain the short page reference
        appended by ``qa.answer_question()``.
    citations:
        Ordered, deduplicated list of citation records from RAG metadata.
    model_used:
        The model name reported by the LLM adapter; empty on failure paths.
    refused:
        True when the model returned the NOT_COVERED refusal.
    spoken_citation:
        Short citation phrase already appended to the spoken ``answer``
        (e.g. ``"You can read that on page 5."``).  Empty when no citation
        exists or when the answer was refused.
    """

    answer: str
    citations: list[CitationRecord] = field(default_factory=list)
    model_used: str = ""
    refused: bool = False
    spoken_citation: str = ""

    def citation_dicts(self) -> list[dict]:
        """Return citations as a list of JSON-safe dicts."""
        return [c.as_dict() for c in self.citations]


def build_spoken_citation(citations: list[CitationRecord]) -> str:
    """Return a short spoken page-reference string from a citation list.

    Examples
    --------
    >>> build_spoken_citation([])
    ''
    >>> build_spoken_citation([CitationRecord(page=5)])
    'You can read that on page 5.'
    >>> build_spoken_citation([CitationRecord(page=2), CitationRecord(page=7)])
    'You can read that on pages 2 and 7.'
    """
    pages = sorted({c.page for c in citations if isinstance(c.page, int) and c.page > 0})
    if not pages:
        return ""
    if len(pages) == 1:
        return f"You can read that on page {pages[0]}."
    return f"You can read that on pages {pages[0]} and {pages[1]}."


def validate_attributed_answer(obj: AttributedAnswer) -> None:
    """Validate the internal consistency of an ``AttributedAnswer``.

    Raises
    ------
    ValueError
        * If ``refused=False``, the answer is not a TROUBLE fallback, and
          ``citations`` is empty — this would mean we are returning an answer
          with no evidence trail.
        * If ``refused=True`` and ``citations`` is non-empty — a refusal must
          not carry page references.
    """
    is_trouble = obj.answer.startswith(_TROUBLE_PREFIX)

    if obj.refused:
        if obj.citations:
            raise ValueError(
                "AttributedAnswer.citations must be empty when refused=True; "
                f"got {len(obj.citations)} citation(s)."
            )
    elif not is_trouble and not obj.citations:
        raise ValueError(
            "AttributedAnswer.citations must be non-empty for a grounded answer "
            "(refused=False and answer is not the TROUBLE fallback). "
            "If the answer is genuinely ungrounded, set refused=True."
        )
