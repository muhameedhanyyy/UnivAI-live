"""Tests for source attribution: CitationRecord, AttributedAnswer, and enrich_citations.

All tests run without LiveKit, LLM, TTS, STT, or database dependencies.
"""

from __future__ import annotations

import os
import unittest

os.environ["UNIVAI_MODE"] = "standalone"

from protocols.source_attribution import (
    AttributedAnswer,
    CitationRecord,
    build_spoken_citation,
    validate_attributed_answer,
)
from citations import enrich_citations


class TestCitationRecord(unittest.TestCase):
    def test_construction_with_page_only(self) -> None:
        c = CitationRecord(page=5)
        self.assertEqual(5, c.page)
        self.assertEqual("", c.programme_id)
        self.assertEqual("", c.chunk_id)

    def test_construction_with_all_fields(self) -> None:
        c = CitationRecord(
            page=12,
            programme_id="prog-001",
            course_id="course-ai",
            lecture_id="lec-wk2",
            plan_version="v3",
            chunk_id="chunk-abc",
        )
        self.assertEqual(12, c.page)
        self.assertEqual("prog-001", c.programme_id)
        self.assertEqual("chunk-abc", c.chunk_id)

    def test_as_dict_contains_required_keys(self) -> None:
        c = CitationRecord(page=7, programme_id="p", course_id="c")
        d = c.as_dict()
        self.assertIn("page", d)
        self.assertIn("programme_id", d)
        self.assertIn("course_id", d)
        self.assertIn("lecture_id", d)
        self.assertIn("plan_version", d)
        self.assertIn("chunk_id", d)
        self.assertEqual(7, d["page"])


class TestBuildSpokenCitation(unittest.TestCase):
    def test_empty_list_returns_empty_string(self) -> None:
        self.assertEqual("", build_spoken_citation([]))

    def test_single_page(self) -> None:
        result = build_spoken_citation([CitationRecord(page=5)])
        self.assertEqual("You can read that on page 5.", result)

    def test_two_pages(self) -> None:
        result = build_spoken_citation([
            CitationRecord(page=2),
            CitationRecord(page=7),
        ])
        self.assertEqual("You can read that on pages 2 and 7.", result)

    def test_three_pages_uses_first_two(self) -> None:
        result = build_spoken_citation([
            CitationRecord(page=1),
            CitationRecord(page=3),
            CitationRecord(page=9),
        ])
        self.assertEqual("You can read that on pages 1 and 3.", result)

    def test_duplicate_pages_deduplicated(self) -> None:
        result = build_spoken_citation([
            CitationRecord(page=4),
            CitationRecord(page=4),
        ])
        self.assertEqual("You can read that on page 4.", result)

    def test_zero_or_negative_page_ignored(self) -> None:
        result = build_spoken_citation([
            CitationRecord(page=0),
            CitationRecord(page=-1),
            CitationRecord(page=3),
        ])
        self.assertEqual("You can read that on page 3.", result)


class TestValidateAttributedAnswer(unittest.TestCase):
    def test_grounded_answer_with_citations_passes(self) -> None:
        obj = AttributedAnswer(
            answer="The answer is X.",
            citations=[CitationRecord(page=2)],
            refused=False,
        )
        # Must not raise
        validate_attributed_answer(obj)

    def test_refused_answer_with_empty_citations_passes(self) -> None:
        obj = AttributedAnswer(
            answer="That is not covered in your book.",
            citations=[],
            refused=True,
        )
        validate_attributed_answer(obj)

    def test_trouble_fallback_with_empty_citations_passes(self) -> None:
        obj = AttributedAnswer(
            answer="I had trouble looking that up. Let me continue.",
            citations=[],
            refused=False,
        )
        validate_attributed_answer(obj)

    def test_grounded_answer_without_citations_raises(self) -> None:
        obj = AttributedAnswer(
            answer="The answer is clearly X.",
            citations=[],
            refused=False,
        )
        with self.assertRaises(ValueError) as ctx:
            validate_attributed_answer(obj)
        self.assertIn("citations must be non-empty", str(ctx.exception))

    def test_refused_with_citations_raises(self) -> None:
        obj = AttributedAnswer(
            answer="That is not covered in your book.",
            citations=[CitationRecord(page=5)],
            refused=True,
        )
        with self.assertRaises(ValueError) as ctx:
            validate_attributed_answer(obj)
        self.assertIn("empty when refused=True", str(ctx.exception))


class TestEnrichCitations(unittest.TestCase):
    SCOPE = {
        "programme_id": "prog-demo",
        "course_id": "course-demo",
        "lecture_id": "lec-demo",
        "plan_version": "v1",
    }

    def test_grounded_result_produces_citations(self) -> None:
        raw = {
            "answer": "Tenant filtering keeps each learner's material separate. You can read that on page 2.",
            "pages": [2],
            "model_used": "test-model",
        }
        result = enrich_citations(raw, **self.SCOPE)
        self.assertFalse(result.refused)
        self.assertEqual(1, len(result.citations))
        self.assertEqual(2, result.citations[0].page)
        self.assertEqual("prog-demo", result.citations[0].programme_id)
        self.assertEqual("test-model", result.model_used)

    def test_refused_answer_produces_empty_citations(self) -> None:
        raw = {
            "answer": "That is not covered in your book, so I cannot answer it from the material.",
            "pages": [],
            "model_used": "test-model",
        }
        result = enrich_citations(raw, **self.SCOPE)
        self.assertTrue(result.refused)
        self.assertEqual([], result.citations)
        self.assertEqual("", result.spoken_citation)

    def test_trouble_fallback_produces_empty_citations(self) -> None:
        raw = {
            "answer": "I had trouble looking that up. Let me continue, and we can come back to it.",
            "pages": [],
            "model_used": "",
        }
        result = enrich_citations(raw, **self.SCOPE)
        self.assertFalse(result.refused)
        self.assertEqual([], result.citations)

    def test_duplicate_pages_are_deduplicated(self) -> None:
        raw = {
            "answer": "Some answer. You can read that on pages 3 and 3.",
            "pages": [3, 3],
            "model_used": "m",
        }
        result = enrich_citations(raw, **self.SCOPE)
        self.assertEqual(1, len(result.citations))
        self.assertEqual(3, result.citations[0].page)

    def test_session_scope_attached_to_each_citation(self) -> None:
        raw = {
            "answer": "Grounded answer. You can read that on pages 1 and 2.",
            "pages": [1, 2],
            "model_used": "m",
        }
        result = enrich_citations(
            raw,
            programme_id="prog-X",
            course_id="course-Y",
            lecture_id="lec-Z",
            plan_version="v9",
        )
        for c in result.citations:
            self.assertEqual("prog-X", c.programme_id)
            self.assertEqual("course-Y", c.course_id)
            self.assertEqual("lec-Z", c.lecture_id)
            self.assertEqual("v9", c.plan_version)

    def test_spoken_citation_matches_pages(self) -> None:
        raw = {
            "answer": "Grounded. You can read that on page 5.",
            "pages": [5],
            "model_used": "m",
        }
        result = enrich_citations(raw, **self.SCOPE)
        self.assertEqual("You can read that on page 5.", result.spoken_citation)

    def test_citation_dicts_are_json_safe(self) -> None:
        import json
        raw = {
            "answer": "Grounded. You can read that on page 7.",
            "pages": [7],
            "model_used": "m",
        }
        result = enrich_citations(raw, **self.SCOPE)
        # Should not raise
        encoded = json.dumps(result.citation_dicts())
        self.assertIn("7", encoded)

    def test_empty_scope_produces_empty_scope_fields(self) -> None:
        raw = {
            "answer": "Grounded. You can read that on page 2.",
            "pages": [2],
            "model_used": "m",
        }
        result = enrich_citations(raw)
        self.assertEqual("", result.citations[0].programme_id)
        self.assertEqual("", result.citations[0].plan_version)


if __name__ == "__main__":
    unittest.main()
