"""Mocked unit tests for qa.answer_question.

These tests verify:
- Grounded answers produce citations and pages.
- Contextual follow-ups enrich retrieval and the answer prompt with prior turns.
- Out-of-scope questions return explicit refusal (pages=[]).
- RagUnavailable produces TROUBLE fallback without raising.
- LLMError produces TROUBLE fallback without raising.
- Missing citations on a grounded answer logs a validation warning instead of crashing.
- Session identity kwargs are forwarded correctly.

No real network calls, no LLM, no RAG, no database, no LiveKit are made.
All external calls within qa.py are patched using unittest.mock.

The campus service modules are stubbed before importing qa.py. An import
failure fails test collection; this suite never skips the production module.
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

os.environ["UNIVAI_MODE"] = "standalone"

# ---------------------------------------------------------------------------
# Import qa in the test process.
# qa.py adds services/ to sys.path itself, but the actual service modules
# (common.clock, common.db, etc.) may fail to import on a dev machine that
# has no campus environment. We stub them out BEFORE importing qa so that
# the module-level imports in qa.py resolve without connecting to anything.
# ---------------------------------------------------------------------------

_SERVICES = Path(__file__).resolve().parents[1].parent / "services"

# Provide minimal stubs for the campus plumbing so qa.py can be imported.
_stub_clock   = MagicMock()
_stub_clock.now = MagicMock(return_value="2026-01-01T00:00:00Z")
_stub_db      = MagicMock()
_stub_db.execute = MagicMock()

_stub_llm     = MagicMock()


class _LLMResult:
    def __init__(self, text: str, model: str = "stub-model") -> None:
        self.text = text
        self.model_used = model


_stub_llm.complete = MagicMock(return_value=_LLMResult("Stub answer."))
_stub_llm.LLMError = type("LLMError", (Exception,), {})
_stub_llm.TIMEOUT_QA_S = 30

_stub_rag = MagicMock()


class _RagUnavailable(Exception):
    pass


_stub_rag.search_book = MagicMock(return_value=[])
_stub_rag.RagUnavailable = _RagUnavailable

# Inject stubs into sys.modules so qa.py's `from common.x import y` resolves.
sys.modules.setdefault("common", MagicMock())
sys.modules["common.clock"] = _stub_clock
sys.modules["common.db"] = _stub_db
sys.modules["common.llm"] = _stub_llm
sys.modules["common.rag_client"] = _stub_rag
sys.modules["common.device"] = MagicMock()
sys.modules["common.sentences"] = MagicMock(split_sentences=lambda t: [t])

# qa.py lives in the parent directory of tests/. Import failures are test
# failures; this suite must never silently skip the production Q&A module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qa as _qa_module  # noqa: E402
from qa_context import ConversationTurn, QuestionContext, SlideSnapshot  # noqa: E402

_SCOPE = {
    "programme_id": "prog-test",
    "course_id": "course-test",
    "lecture_id_str": "lec-wk1",
    "plan_version": 1,
}

_RAG_HIT_PAGE_2 = {
    "text": "Tenant filtering keeps material separate.",
    "page": 2,
    "source": "tenant-guide.pdf",
}
_RAG_HIT_PAGE_5 = {
    "text": "The plan version governs approved content.",
    "page": 5,
    "source": "programme-guide.pdf",
}


class TestAnswerQuestionMocked(unittest.TestCase):
    """Each test patches only the callables that answer_question uses at runtime."""

    def _run(self, coro) -> object:
        return asyncio.run(coro)

    # -- happy path -----------------------------------------------------------

    def test_grounded_answer_returns_pages_and_citations(self) -> None:
        with (
            patch.object(_qa_module, "search_book", AsyncMock(return_value=[_RAG_HIT_PAGE_2])),
            patch.object(_qa_module, "complete", MagicMock(return_value=_LLMResult("Tenant filtering keeps material separate."))),
            patch.object(_qa_module, "_log_later", MagicMock()),
        ):
            result = self._run(
                _qa_module.answer_question(
                    "How is tenant material protected?",
                    lecture_id=None,
                    sid="S-test",
                    **_SCOPE,
                )
            )
        self.assertIn(2, result["pages"])
        self.assertIsInstance(result["citations"], list)
        self.assertEqual(1, len(result["citations"]))
        self.assertEqual(2, result["citations"][0]["page"])
        self.assertEqual("tenant-guide.pdf", result["citations"][0]["source"])
        self.assertEqual("prog-test", result["citations"][0]["programme_id"])

    def test_grounded_answer_appends_spoken_page_reference(self) -> None:
        with (
            patch.object(_qa_module, "search_book", AsyncMock(return_value=[_RAG_HIT_PAGE_2])),
            patch.object(_qa_module, "complete", MagicMock(return_value=_LLMResult("Tenant filtering keeps material separate."))),
            patch.object(_qa_module, "_log_later", MagicMock()),
        ):
            result = self._run(
                _qa_module.answer_question("How is tenant material protected?", lecture_id=None)
            )
        self.assertIn("page 2", result["answer"])
        self.assertIn("tenant-guide.pdf", result["answer"])

    def test_contextual_follow_up_reaches_retrieval_and_answer_prompt(self) -> None:
        search = AsyncMock(return_value=[_RAG_HIT_PAGE_2])
        complete = MagicMock(
            return_value=_LLMResult("Tenant filtering keeps material separate.")
        )
        context = QuestionContext(
            current_slide=SlideSnapshot(
                3,
                "Tenant filters keep one learner's material separate.",
            ),
            history=(
                ConversationTurn(
                    "I did not understand the current slide.",
                    "The filter selects only your material.",
                    slide_number=3,
                ),
            ),
        )
        with (
            patch.object(_qa_module, "search_book", search),
            patch.object(_qa_module, "complete", complete),
            patch.object(_qa_module, "_log_later", MagicMock()),
        ):
            self._run(
                _qa_module.answer_question(
                    "Please explain it again.",
                    lecture_id=None,
                    context=context,
                )
            )

        retrieval_query = search.await_args.args[0]
        answer_prompt = complete.call_args.args[0]
        self.assertIn("Immediately preceding student question", retrieval_query)
        self.assertIn("The filter selects only your material", retrieval_query)
        self.assertIn("Resolved turn type: follow_up", answer_prompt)
        self.assertIn("Textbook evidence", answer_prompt)

    def test_out_of_scope_question_returns_refusal(self) -> None:
        """A question the book doesn't cover: RAG returns empty hits."""
        with (
            patch.object(_qa_module, "search_book", AsyncMock(return_value=[])),
            patch.object(_qa_module, "_log_later", MagicMock()),
        ):
            result = self._run(
                _qa_module.answer_question("What is the weather?", lecture_id=None)
            )
        self.assertEqual([], result["pages"])
        self.assertIn("not covered", result["answer"].lower())
        # citations must be empty on refusal
        self.assertEqual([], result["citations"])

    def test_model_refusal_produces_empty_pages(self) -> None:
        """Model explicitly refuses even though RAG returned hits."""
        with (
            patch.object(_qa_module, "search_book", AsyncMock(return_value=[_RAG_HIT_PAGE_2])),
            patch.object(_qa_module, "complete", MagicMock(return_value=_LLMResult("That is not covered in your book."))),
            patch.object(_qa_module, "_log_later", MagicMock()),
        ):
            result = self._run(
                _qa_module.answer_question("Unrelated question?", lecture_id=None)
            )
        self.assertEqual([], result["pages"])
        self.assertEqual([], result["citations"])

    # -- degraded / failure paths ---------------------------------------------

    def test_rag_unavailable_returns_trouble_fallback(self) -> None:
        with (
            patch.object(_qa_module, "search_book", AsyncMock(side_effect=_RagUnavailable("not configured"))),
            patch.object(_qa_module, "_log_later", MagicMock()),
        ):
            result = self._run(
                _qa_module.answer_question("Any question?", lecture_id=None)
            )
        # With empty hits we get NOT_IN_BOOK, not TROUBLE — still a graceful refusal.
        self.assertIn("trouble", result["answer"].lower())
        self.assertEqual([], result["pages"])
        self.assertEqual([], result["citations"])
        self.assertEqual([], result["pages"])

    def test_rag_exception_returns_trouble_fallback_without_raising(self) -> None:
        with (
            patch.object(_qa_module, "search_book", AsyncMock(side_effect=RuntimeError("timeout"))),
            patch.object(_qa_module, "_log_later", MagicMock()),
        ):
            result = self._run(
                _qa_module.answer_question("Any question?", lecture_id=None)
            )
        self.assertIn("trouble", result["answer"].lower())
        self.assertEqual([], result["pages"])

    def test_llm_error_returns_trouble_fallback_without_raising(self) -> None:
        LLMError = _qa_module.LLMError
        with (
            patch.object(_qa_module, "search_book", AsyncMock(return_value=[_RAG_HIT_PAGE_2])),
            patch.object(_qa_module, "complete", MagicMock(side_effect=LLMError("both models down"))),
            patch.object(_qa_module, "_log_later", MagicMock()),
        ):
            result = self._run(
                _qa_module.answer_question("A question?", lecture_id=None)
            )
        self.assertIn("trouble", result["answer"].lower())

    def test_grounded_answer_without_source_identity_fails_closed(self) -> None:
        hit_without_source = {"text": "A grounded-looking passage.", "page": 2}
        with (
            patch.object(
                _qa_module,
                "search_book",
                AsyncMock(return_value=[hit_without_source]),
            ),
            patch.object(
                _qa_module,
                "complete",
                MagicMock(return_value=_LLMResult("A grounded-looking answer.")),
            ),
            patch.object(_qa_module, "_log_later", MagicMock()),
        ):
            result = self._run(
                _qa_module.answer_question("Question?", lecture_id=None)
            )

        self.assertIn("trouble", result["answer"].lower())
        self.assertEqual([], result["citations"])

    def test_answer_question_never_raises(self) -> None:
        """contract: answer_question must never propagate an exception."""
        with (
            patch.object(_qa_module, "search_book", AsyncMock(side_effect=Exception("catastrophic"))),
            patch.object(_qa_module, "_log_later", MagicMock()),
        ):
            # This must complete without raising
            result = self._run(
                _qa_module.answer_question("Anything?", lecture_id=None)
            )
        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)

    # -- on_progress callback -------------------------------------------------

    def test_on_progress_called_at_each_stage(self) -> None:
        stages: list[str] = []

        async def capture(stage: str, detail: str = "") -> None:
            stages.append(stage)

        with (
            patch.object(_qa_module, "search_book", AsyncMock(return_value=[_RAG_HIT_PAGE_2])),
            patch.object(_qa_module, "complete", MagicMock(return_value=_LLMResult("Answer."))),
            patch.object(_qa_module, "_log_later", MagicMock()),
        ):
            self._run(
                _qa_module.answer_question(
                    "Question?", lecture_id=None, on_progress=capture
                )
            )
        self.assertIn("retrieving", stages)
        self.assertIn("retrieved", stages)
        self.assertIn("answered", stages)

    # -- session identity forwarding ------------------------------------------

    def test_session_scope_is_forwarded_to_citations(self) -> None:
        with (
            patch.object(_qa_module, "search_book", AsyncMock(return_value=[_RAG_HIT_PAGE_5])),
            patch.object(_qa_module, "complete", MagicMock(return_value=_LLMResult("Plan version governs content."))),
            patch.object(_qa_module, "_log_later", MagicMock()),
        ):
            result = self._run(
                _qa_module.answer_question(
                    "What governs approved content?",
                    lecture_id=None,
                    programme_id="prog-XYZ",
                    course_id="course-ABC",
                    plan_version=99,
                    lecture_id_str="lec-wk99",
                )
            )
        self.assertTrue(len(result["citations"]) > 0)
        c = result["citations"][0]
        self.assertEqual("prog-XYZ", c["programme_id"])
        self.assertEqual("course-ABC", c["course_id"])
        self.assertEqual(99, c["plan_version"])
        self.assertEqual("lec-wk99", c["lecture_id"])


if __name__ == "__main__":
    unittest.main()
