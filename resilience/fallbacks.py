"""Explicit, learner-visible fallback decisions."""

from dataclasses import asdict, dataclass
from enum import Enum


class FallbackMode(str, Enum):
    TEXT_QUESTION = "text_question"
    TEXT_ANSWER = "text_answer"
    DEFERRED_ANSWER = "deferred_answer"


@dataclass(frozen=True)
class FallbackDecision:
    mode: FallbackMode
    reason_code: str
    learner_message: str
    preserves_lecture: bool = True

    def event(self) -> dict:
        return {"type": "fallback", "payload": asdict(self)}


def choose_fallback(stage: str, reason_code: str) -> FallbackDecision:
    if stage == "stt":
        return FallbackDecision(FallbackMode.TEXT_QUESTION, reason_code, "Voice capture is unavailable. Please type your question.")
    if stage == "tts":
        return FallbackDecision(FallbackMode.TEXT_ANSWER, reason_code, "Voice playback is unavailable. Your grounded answer is shown as text.")
    return FallbackDecision(FallbackMode.DEFERRED_ANSWER, reason_code, "The answer service is delayed. Your question was deferred and the lecture will continue.")
