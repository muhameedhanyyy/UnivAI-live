"""Bounded conversational context for raised-hand lecture questions.

The lecture worker owns one :class:`ConversationMemory` per learner room.  It
captures enough context to resolve phrases such as "the previous slide" and
"explain it again" without turning earlier model output into evidence.  The
Q&A pipeline still retrieves textbook passages for every factual answer.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence


MAX_HISTORY_TURNS = 6
MAX_SLIDE_CHARS = 2_000
MAX_HISTORY_QUESTION_CHARS = 500
MAX_HISTORY_ANSWER_CHARS = 1_200
MAX_RETRIEVAL_ANSWER_CHARS = 600


class QuestionIntent(str, Enum):
    """The contextual target of the learner's latest message."""

    STANDALONE = "standalone"
    CURRENT_SLIDE = "current_slide"
    PREVIOUS_SLIDE = "previous_slide"
    FOLLOW_UP = "follow_up"


@dataclass(frozen=True)
class SlideSnapshot:
    number: int
    text: str


@dataclass(frozen=True)
class ConversationTurn:
    question: str
    answer: str
    slide_number: int | None = None


@dataclass(frozen=True)
class QuestionContext:
    current_slide: SlideSnapshot | None = None
    previous_slide: SlideSnapshot | None = None
    history: tuple[ConversationTurn, ...] = ()


def context_to_dict(context: QuestionContext) -> dict:
    """Serialize only the bounded fields accepted by :func:`context_from_dict`."""

    def slide(value: SlideSnapshot | None) -> dict | None:
        return None if value is None else {"number": value.number, "text": value.text}

    return {
        "current_slide": slide(context.current_slide),
        "previous_slide": slide(context.previous_slide),
        "history": [
            {
                "question": turn.question,
                "answer": turn.answer,
                "slide_number": turn.slide_number,
            }
            for turn in context.history[-MAX_HISTORY_TURNS:]
        ],
    }


def context_from_dict(value: object) -> QuestionContext:
    """Rebuild a safe bounded context snapshot stored with a prior answer."""

    data = value if isinstance(value, dict) else {}

    def slide(raw: object) -> SlideSnapshot | None:
        if not isinstance(raw, dict):
            return None
        number = raw.get("number")
        text = _clip(raw.get("text"), MAX_SLIDE_CHARS)
        if not isinstance(number, int) or number < 1 or not text:
            return None
        return SlideSnapshot(number=number, text=text)

    turns: list[ConversationTurn] = []
    raw_history = data.get("history")
    if isinstance(raw_history, list):
        for raw in raw_history[-MAX_HISTORY_TURNS:]:
            if not isinstance(raw, dict):
                continue
            question = _clip(raw.get("question"), MAX_HISTORY_QUESTION_CHARS)
            answer = _clip(raw.get("answer"), MAX_HISTORY_ANSWER_CHARS)
            slide_number = raw.get("slide_number")
            if not question or not answer:
                continue
            turns.append(
                ConversationTurn(
                    question=question,
                    answer=answer,
                    slide_number=slide_number
                    if isinstance(slide_number, int) and slide_number > 0
                    else None,
                )
            )
    return QuestionContext(
        current_slide=slide(data.get("current_slide")),
        previous_slide=slide(data.get("previous_slide")),
        history=tuple(turns),
    )


_PREVIOUS_SLIDE = re.compile(
    r"\b(?:previous|prior|last)\s+(?:slide|page)\b|"
    r"\b(?:slide|page)\s+(?:before|back)\b|"
    r"(?:الشريحة|السلايد)\s+(?:السابقة|السابق|اللي\s+فات)",
    re.IGNORECASE,
)
_CURRENT_SLIDE = re.compile(
    r"\b(?:current|this)\s+(?:slide|page)\b|"
    r"\b(?:slide|page)\s+(?:we(?:'| a)?re\s+on|now)\b|"
    r"(?:الشريحة|السلايد)\s+(?:الحالية|الحالي|دي|ده)",
    re.IGNORECASE,
)
_FOLLOW_UP = re.compile(
    r"\b(?:again|it|that|those|them|same|simpler|differently|more|"
    r"didn['’]?t\s+understand|did\s+not\s+understand|don['’]?t\s+get|"
    r"still\s+confused|what\s+do\s+you\s+mean)\b|"
    r"(?:اشرح(?:ها|ه)?\s+تاني|مرة\s+أخرى|مش\s+فاهم|مش\s+فاهمة|"
    r"لم\s+أفهم|ما\s+فهمت)",
    re.IGNORECASE,
)
_CURRENT_CONFUSION = re.compile(
    r"\b(?:i\s+)?(?:do\s+not|don['’]?t|did\s+not|didn['’]?t)\s+understand\b|"
    r"\b(?:i(?:'| a)?m\s+)?confused\b|"
    r"(?:مش\s+فاهم|مش\s+فاهمة|لم\s+أفهم|ما\s+فهمت)",
    re.IGNORECASE,
)
_BARE_FOLLOW_UP = re.compile(
    r"^(?:why|how|how so|what do you mean|can you explain|explain more)[?.!\s]*$",
    re.IGNORECASE,
)


def _compact(value: object) -> str:
    return re.sub(r"\s+", " ", value if isinstance(value, str) else "").strip()


def _clip(value: object, limit: int) -> str:
    compact = _compact(value)
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def classify_question(question: str, context: QuestionContext) -> QuestionIntent:
    """Resolve which nearby conversational object the question targets."""

    normalized = _compact(question)
    if _PREVIOUS_SLIDE.search(normalized):
        return QuestionIntent.PREVIOUS_SLIDE
    if _CURRENT_SLIDE.search(normalized):
        return QuestionIntent.CURRENT_SLIDE
    if context.history and (
        _FOLLOW_UP.search(normalized) or _BARE_FOLLOW_UP.fullmatch(normalized)
    ):
        return QuestionIntent.FOLLOW_UP
    if _CURRENT_CONFUSION.search(normalized):
        return QuestionIntent.CURRENT_SLIDE
    return QuestionIntent.STANDALONE


def build_retrieval_query(question: str, context: QuestionContext) -> str:
    """Make an ambiguous follow-up searchable without an extra model call.

    Direct, self-contained questions are returned unchanged.  Context is added
    only when the wording points at a slide or an earlier exchange, which keeps
    ordinary retrieval focused and makes contextual turns deterministic.
    """

    latest = _clip(question, MAX_HISTORY_QUESTION_CHARS)
    intent = classify_question(latest, context)
    parts = [latest]

    if intent is QuestionIntent.PREVIOUS_SLIDE and context.previous_slide:
        parts.append(
            f"Previous lecture slide {context.previous_slide.number}: "
            f"{context.previous_slide.text}"
        )
    elif intent is QuestionIntent.CURRENT_SLIDE and context.current_slide:
        parts.append(
            f"Current lecture slide {context.current_slide.number}: "
            f"{context.current_slide.text}"
        )
    elif intent is QuestionIntent.FOLLOW_UP and context.history:
        previous = context.history[-1]
        parts.append(f"Immediately preceding student question: {previous.question}")
        parts.append(
            "Immediately preceding grounded answer: "
            f"{_clip(previous.answer, MAX_RETRIEVAL_ANSWER_CHARS)}"
        )
        referenced_slide = _slide_for_turn(previous, context)
        if referenced_slide:
            parts.append(
                f"Referenced lecture slide {referenced_slide.number}: "
                f"{referenced_slide.text}"
            )

    return "\n".join(part for part in parts if part)


def build_answer_prompt(
    question: str,
    passages: Sequence[str],
    context: QuestionContext,
) -> str:
    """Render the latest turn, bounded chat history, and textbook evidence."""

    intent = classify_question(question, context)
    position_lines: list[str] = []
    if context.current_slide:
        position_lines.append(
            f"Current slide {context.current_slide.number}: {context.current_slide.text}"
        )
    if context.previous_slide:
        position_lines.append(
            f"Previous slide {context.previous_slide.number}: {context.previous_slide.text}"
        )

    history_lines: list[str] = []
    for index, turn in enumerate(context.history, start=1):
        slide = f" on slide {turn.slide_number}" if turn.slide_number else ""
        history_lines.extend(
            (
                f"Turn {index} student{slide}: {turn.question}",
                f"Turn {index} assistant: {turn.answer}",
            )
        )

    return (
        f"Latest student message: {question}\n"
        f"Resolved turn type: {intent.value}\n\n"
        "Lecture position (reference context only; not factual evidence):\n"
        + ("\n".join(position_lines) if position_lines else "No slide context available.")
        + "\n\nRecent conversation (reference context only; not factual evidence):\n"
        + ("\n".join(history_lines) if history_lines else "No earlier turns.")
        + "\n\nTextbook evidence (the only source for factual claims):\n"
        + "\n\n".join(passages)
        + "\n\nAnswer the latest message as a natural continuation. Resolve words like "
        "'it', 'that', 'again', 'current slide', and 'previous slide' from the "
        "reference context. If the student asks for another explanation, explain "
        "the same idea differently with simpler wording or one concrete example; "
        "do not merely repeat the prior answer. Give the conclusion and a concise "
        "explanation of the key connection, but never reveal private chain-of-thought. "
        "Use only the textbook evidence for factual claims and answer in at most four "
        "short spoken sentences."
    )


class ConversationMemory:
    """Per-room, bounded memory for a live learner/lecturer conversation."""

    def __init__(
        self,
        segments: Sequence[dict],
        *,
        max_turns: int = MAX_HISTORY_TURNS,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self._slides = _slide_snapshots(segments)
        self._segment_slides = tuple(
            segment.get("slide") if isinstance(segment.get("slide"), int) else None
            for segment in segments
        )
        self._turns: deque[ConversationTurn] = deque(maxlen=max_turns)

    def context_at(self, segment_index: int) -> QuestionContext:
        current_number = (
            self._segment_slides[segment_index]
            if 0 <= segment_index < len(self._segment_slides)
            else None
        )
        current = self._slides.get(current_number) if current_number else None
        previous = None
        if current_number is not None:
            ordered = list(self._slides)
            try:
                current_position = ordered.index(current_number)
            except ValueError:
                current_position = -1
            if current_position > 0:
                previous = self._slides[ordered[current_position - 1]]
        return QuestionContext(
            current_slide=current,
            previous_slide=previous,
            history=tuple(self._turns),
        )

    def record(
        self,
        question: str,
        answer: str,
        *,
        slide_number: int | None,
    ) -> None:
        self._turns.append(
            ConversationTurn(
                question=_clip(question, MAX_HISTORY_QUESTION_CHARS),
                answer=_clip(answer, MAX_HISTORY_ANSWER_CHARS),
                slide_number=slide_number,
            )
        )

    @property
    def turns(self) -> tuple[ConversationTurn, ...]:
        return tuple(self._turns)


def _slide_for_turn(
    turn: ConversationTurn,
    context: QuestionContext,
) -> SlideSnapshot | None:
    for slide in (context.current_slide, context.previous_slide):
        if slide and slide.number == turn.slide_number:
            return slide
    return None


def _slide_snapshots(segments: Iterable[dict]) -> dict[int, SlideSnapshot]:
    grouped: dict[int, list[str]] = {}
    for segment in segments:
        number = segment.get("slide")
        text = _compact(segment.get("text"))
        if not isinstance(number, int) or number < 1 or not text:
            continue
        grouped.setdefault(number, []).append(text)
    return {
        number: SlideSnapshot(number, _clip(" ".join(texts), MAX_SLIDE_CHARS))
        for number, texts in grouped.items()
    }


__all__ = [
    "ConversationMemory",
    "ConversationTurn",
    "QuestionContext",
    "QuestionIntent",
    "SlideSnapshot",
    "build_answer_prompt",
    "build_retrieval_query",
    "classify_question",
    "context_from_dict",
    "context_to_dict",
]
