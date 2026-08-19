"""Deterministic tests for contextual raised-hand questions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa_context import (
    ConversationMemory,
    QuestionIntent,
    build_answer_prompt,
    build_retrieval_query,
    classify_question,
    resolve_slide_reference,
)


SEGMENTS = [
    {"slide": 1, "text": "Welcome to hashing."},
    {"slide": 2, "text": "A hash function maps a key to a bucket."},
    {"slide": 2, "text": "Collisions occur when keys map to one bucket."},
    {"slide": 3, "text": "Chaining stores colliding keys in a bucket list."},
]


def test_slide_context_combines_segments_and_selects_previous_slide():
    memory = ConversationMemory(SEGMENTS)

    context = memory.context_at(3)

    assert context.current_slide is not None
    assert context.current_slide.number == 3
    assert "Chaining" in context.current_slide.text
    assert context.previous_slide is not None
    assert context.previous_slide.number == 2
    assert "hash function" in context.previous_slide.text
    assert "Collisions" in context.previous_slide.text


def test_previous_slide_request_retrieves_against_previous_slide_text():
    context = ConversationMemory(SEGMENTS).context_at(3)

    query = build_retrieval_query("Please repeat the previous slide", context)

    assert classify_question("Please repeat the previous slide", context) is QuestionIntent.PREVIOUS_SLIDE
    assert "Previous lecture slide 2" in query
    assert "hash function" in query
    assert "Chaining stores" not in query


def test_current_slide_confusion_retrieves_against_current_slide_text():
    context = ConversationMemory(SEGMENTS).context_at(3)

    query = build_retrieval_query("I didn't understand the current slide", context)

    assert classify_question("I didn't understand the current slide", context) is QuestionIntent.CURRENT_SLIDE
    assert "Current lecture slide 3" in query
    assert "Chaining stores" in query


def test_follow_up_uses_immediately_preceding_exchange_and_slide():
    memory = ConversationMemory(SEGMENTS)
    memory.record(
        "I didn't understand the current slide",
        "Chaining puts colliding keys into a list for that bucket.",
        slide_number=3,
    )
    context = memory.context_at(3)

    query = build_retrieval_query("I didn't understand; explain it again please", context)

    assert classify_question("I didn't understand; explain it again please", context) is QuestionIntent.FOLLOW_UP
    assert "Immediately preceding student question" in query
    assert "Chaining puts colliding keys" in query
    assert "Referenced lecture slide 3" in query


def test_direct_question_stays_focused_for_retrieval():
    memory = ConversationMemory(SEGMENTS)
    memory.record("What is a collision?", "Two keys share a bucket.", slide_number=2)
    context = memory.context_at(3)

    assert build_retrieval_query("What is a load factor?", context) == "What is a load factor?"


def test_prompt_separates_reference_context_from_textbook_evidence():
    memory = ConversationMemory(SEGMENTS)
    memory.record("What is chaining?", "It uses a bucket list.", slide_number=3)
    context = memory.context_at(3)

    prompt = build_answer_prompt(
        "Can you explain it again?",
        ["[page 12] Chaining keeps a linked list at each occupied slot."],
        context,
    )

    assert "Resolved turn type: follow_up" in prompt
    assert "reference context only; not factual evidence" in prompt
    assert "the only source for factual claims" in prompt
    assert "do not merely repeat the prior answer" in prompt
    assert "never reveal private chain-of-thought" in prompt


def test_memory_is_bounded_to_recent_turns():
    memory = ConversationMemory(SEGMENTS, max_turns=2)
    memory.record("first", "one", slide_number=1)
    memory.record("second", "two", slide_number=2)
    memory.record("third", "three", slide_number=3)

    assert [turn.question for turn in memory.turns] == ["second", "third"]


def test_explicit_slide_reference_resolves_only_existing_slides():
    context = ConversationMemory(SEGMENTS).context_at(3)

    assert resolve_slide_reference("Can you explain slide 1?", context, [1, 2, 3]) == 1
    assert resolve_slide_reference("اشرح السلايد رقم ٢", context, [1, 2, 3]) == 2
    assert resolve_slide_reference("What is on slide 99?", context, [1, 2, 3]) is None


def test_relative_and_follow_up_slide_references_resolve_deterministically():
    memory = ConversationMemory(SEGMENTS)
    memory.record("Explain slide 2", "A hash maps a key to a bucket.", slide_number=2)
    context = memory.context_at(3)

    assert resolve_slide_reference("Show the previous slide", context, [1, 2, 3]) == 2
    assert resolve_slide_reference("Explain it again", context, [1, 2, 3]) == 2
    assert resolve_slide_reference("Show the next slide", context, [1, 2, 3]) is None
