"""Canonical App/Live message and lecture-script contract."""

from __future__ import annotations

STATES = {
    "connecting",
    "preparing",
    "lecturing",
    "asking",
    "listening",
    "processing",
    "review",
    "answering",
    "ended",
}
SPEECH_STATES = {"waiting", "detected", "processing", "received", "no_speech", "error"}
INBOUND_TYPES = {"raise_hand", "mic", "question", "retry", "cancel"}
OUTBOUND_TYPES = {"slide", "state", "answer", "transcript", "progress", "hand", "speech", "fallback"}


def parse_room_name(room_name: str) -> tuple[str, int]:
    prefix, separator, week_value = room_name.rpartition("-week-")
    if not separator or not prefix.startswith("lecture-"):
        raise ValueError(f"unexpected room name: {room_name}")
    student_id = prefix[len("lecture-") :]
    if not student_id:
        raise ValueError("room name is missing student identity")
    if not week_value.isascii() or not week_value.isdecimal():
        raise ValueError("room week must be a positive integer")
    week = int(week_value)
    if week < 1:
        raise ValueError("room week must be a positive integer")
    return student_id, week


def validate_script(data: dict) -> None:
    if not isinstance(data.get("lectureId"), str) or not data["lectureId"]:
        raise ValueError("lectureId is required")
    if not isinstance(data.get("title"), str) or not data["title"]:
        raise ValueError("title is required")
    if not isinstance(data.get("segments"), list) or not data["segments"]:
        raise ValueError("segments must be a non-empty list")
    previous = 0
    for segment in data["segments"]:
        slide = segment.get("slide")
        if not isinstance(slide, int) or slide < 1 or slide < previous:
            raise ValueError("segment slides must be positive and ordered")
        previous = slide
        if not isinstance(segment.get("text"), str) or not segment["text"].strip():
            raise ValueError("segment text is required")
        citations = segment.get("citations")
        if not isinstance(citations, list) or not citations:
            raise ValueError("segment citations are required")


def validate_inbound(message: dict) -> None:
    kind = message.get("type")
    if kind not in INBOUND_TYPES:
        raise ValueError(f"unknown inbound message type: {kind}")
    if kind == "mic" and not isinstance(message.get("muted"), bool):
        raise ValueError("mic.muted must be boolean")
    if kind == "question":
        text = message.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > 500:
            raise ValueError("question.text must contain 1..500 characters")


def validate_outbound(message: dict) -> None:
    kind = message.get("type")
    if kind not in OUTBOUND_TYPES:
        raise ValueError(f"unknown outbound message type: {kind}")
    if kind == "state" and message.get("state") not in STATES:
        raise ValueError(f"unknown lecture state: {message.get('state')}")
    if kind == "slide" and (
        not isinstance(message.get("n"), int) or message["n"] < 1
    ):
        raise ValueError("slide.n must be a positive integer")
    if kind == "hand" and message.get("state") not in {"acked", "lowered"}:
        raise ValueError("hand.state must be acked or lowered")
    if kind == "speech":
        if message.get("state") not in SPEECH_STATES:
            raise ValueError("unknown speech state")
        detail = message.get("detail")
        if detail is not None and (not isinstance(detail, str) or len(detail) > 300):
            raise ValueError("speech.detail must contain at most 300 characters")
