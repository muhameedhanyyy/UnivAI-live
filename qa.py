"""Answering a student's spoken question during a lecture.

Path: STT text -> the team's RAG (over MCP) -> tiny LLM -> short spoken answer.

The one rule that cannot bend: if RAG returns nothing, we say the book does not
cover it. We never invent an answer. Everything is logged to qa_log with the
model that actually served it.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services"))  # campus plumbing

from common.clock import now  # noqa: E402
from common.db import execute  # noqa: E402
from common.llm import complete, LLMError, TIMEOUT_QA_S  # noqa: E402
from common.rag_client import search_book, RagUnavailable  # noqa: E402
from citations import enrich_citations  # noqa: E402

# Three short spoken sentences are ~60 tokens. The old uncapped call let the
# model ramble to its 180-token default — well over a minute of SPOKEN speech
# the student then sat through (measured: a 120-token answer = 25s of audio;
# the complaint "the speak takes 4:35" was mostly the answer's own length).
# Cap hard; the prompt already demands brevity.
ANSWER_MAX_TOKENS = 90

# Their RAG returns ~3x the passages asked for (observed: top_k=5 -> 15 hits).
# Passing all of them triples the 3B model's prompt for no answer-quality gain
# and slows prefill; the reranker already ordered them, so keep the best few.
MAX_PASSAGES = 5

# The RAG service always returns nearest neighbours — even for a question the book
# does not cover, a vector search still hands back its closest chunks. So an empty
# result is NOT how we detect "not in the book": the model has to refuse when the
# passages it was given do not actually answer the question. Hence the blunt prompt.
#
# The model must NOT speak page numbers either: asked to, a small model invents them
# (observed: it said "page 4" for a passage that came from page 2). The true page comes
# from the RAG metadata and we append it ourselves, below.
SYSTEM = (
    "You are a university teaching assistant answering a student mid-lecture. "
    "Use ONLY the textbook passages given to you. Never add outside knowledge. "
    "The passages are the closest matches found, and they may be irrelevant: if they "
    "do not actually answer the question, reply exactly 'That is not covered in your "
    "book.' and nothing else. Otherwise answer in at most three short spoken sentences. "
    "Never state a page or chunk number — the page reference is added for you."
)

NOT_COVERED = "that is not covered in your book"

NOT_IN_BOOK = (
    "That is not covered in your book, so I cannot answer it from the material. "
    "Let us stay with what the text says."
)

TROUBLE = "I had trouble looking that up. Let me continue, and we can come back to it."

# Fire-and-forget qa_log writes. The INSERT (plus the virtual-clock read inside
# it) opens Postgres connections; under load that took seconds ON THE EVENT
# LOOP, freezing the room right between "answered" and the first spoken word.
_LOG_TASKS: set = set()


def _log_later(
    lecture_id: int | None, sid: str | None, question: str, answer: str, pages: list[int], model: str
) -> None:
    task = asyncio.create_task(
        asyncio.to_thread(_log, lecture_id, sid, question, answer, pages, model)
    )
    _LOG_TASKS.add(task)
    task.add_done_callback(_LOG_TASKS.discard)


async def answer_question(
    question: str,
    lecture_id: int | None,
    sid: str | None = None,
    on_progress=None,
    *,
    programme_id: str = "",
    course_id: str = "",
    plan_version: str = "",
    lecture_id_str: str = "",
) -> dict:
    """Returns {answer, pages, model_used, citations}. Never raises: the lecture must go on.

    sid (the student's studentId) scopes RAG retrieval to THEIR book and stamps
    the qa_log row. on_progress(stage, detail) is awaited at each step so the
    browser can show WHERE the answer currently is instead of looking frozen.

    programme_id, course_id, plan_version, lecture_id_str carry the session
    identity from LectureSessionMeta and are attached to each citation record.
    They default to empty string for backward-compatible callers."""

    async def progress(stage: str, detail: str = "") -> None:
        if on_progress:
            await on_progress(stage, detail)

    pages: list[int] = []
    model_used = ""
    started = time.perf_counter()

    try:
        await progress("retrieving", "")
        hits = await search_book(question, top_k=5, user_id=sid)
        await progress(
            "retrieved",
            f"{len(hits)} passages in {time.perf_counter() - started:.1f}s",
        )
    except RagUnavailable as exc:
        print(f"[qa] RAG not configured: {exc}")
        await progress("problem", f"book search unavailable ({exc})")
        hits = []
    except Exception as exc:
        print(f"[qa] RAG failed: {exc}")
        await progress("problem", "book search failed - apologising and moving on")
        _log_later(lecture_id, sid, question, TROUBLE, [], "")
        return {"answer": TROUBLE, "pages": [], "model_used": "", "citations": []}

    if not hits:
        _log_later(lecture_id, sid, question, NOT_IN_BOOK, [], "")
        return {"answer": NOT_IN_BOOK, "pages": [], "model_used": "", "citations": []}

    # Their reranker can hand back the same chunk twice; feeding duplicates to a small
    # model just wastes its context.
    passages = []
    seen: set[str] = set()
    for hit in hits:
        if len(passages) >= MAX_PASSAGES:
            break
        text = (hit.get("text") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        page = hit.get("page")
        if isinstance(page, int):
            pages.append(page)
        passages.append(f"[page {page}] {text}")

    prompt = (
        f"Student's question: {question}\n\n"
        "Textbook passages:\n" + "\n\n".join(passages) + "\n\n"
        "Answer the question using only these passages, in at most three spoken sentences."
    )

    llm_started = time.perf_counter()
    try:
        import os
        await progress("thinking", f"asking {os.getenv('LLM_PRIMARY', 'the model')}")
        # complete() is synchronous urllib; on the event loop it would freeze the
        # room (no audio, no data messages) for the whole generation. Keep the
        # QA timeout even with the cap set (a cap normally means "generation").
        result = await asyncio.to_thread(
            complete, prompt, SYSTEM, ANSWER_MAX_TOKENS, None, TIMEOUT_QA_S
        )
        answer, model_used = result.text.strip(), result.model_used
        await progress("answered", f"{model_used} in {time.perf_counter() - llm_started:.1f}s")
    except LLMError as exc:
        # Both primary and fallback are down. Say something graceful and keep lecturing.
        print(f"[qa] all models failed: {exc}")
        await progress("problem", "both models failed - apologising and moving on")
        answer = TROUBLE

    cited = sorted(set(pages))

    # The page reference is OURS, taken from the RAG metadata — never the model's word.
    refused = NOT_COVERED in answer.lower()
    if cited and not refused and answer != TROUBLE:
        where = f"page {cited[0]}" if len(cited) == 1 else f"pages {cited[0]} and {cited[1]}"
        answer = f"{answer.rstrip('.')}. You can read that on {where}."
    if refused:
        cited = []

    _log_later(lecture_id, sid, question, answer, cited, model_used)

    # Enrich with typed citation records carrying session identity.
    # enrich_citations() validates the result; on an internal inconsistency it
    # raises ValueError — caught here so a pipeline bug never crashes the room.
    try:
        attributed = enrich_citations(
            {"answer": answer, "pages": cited, "model_used": model_used},
            programme_id=programme_id,
            course_id=course_id,
            lecture_id=lecture_id_str,
            plan_version=plan_version,
        )
        citations_payload = attributed.citation_dicts()
    except ValueError as exc:
        print(f"[qa] citation enrichment validation error: {exc}")
        citations_payload = [{"page": p} for p in cited]

    return {"answer": answer, "pages": cited, "model_used": model_used, "citations": citations_payload}


def _log(
    lecture_id: int | None, sid: str | None, question: str, answer: str, pages: list[int], model: str
) -> None:
    import json

    try:
        execute(
            "INSERT INTO qa_log (student_id, lecture_id, question, answer, citations, model_used, asked_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                sid,
                lecture_id,
                question,
                answer,
                json.dumps([{"page": p} for p in pages]),
                model or None,
                now(),
            ),
        )
    except Exception as exc:  # a dead qa_log must never take the lecture with it
        print(f"[qa] qa_log write failed: {exc}")
