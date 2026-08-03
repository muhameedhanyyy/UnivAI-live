"""Turn-scoped endpointing for complete, learner-confirmed questions."""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable


class TurnState(str, Enum):
    IDLE = "idle"
    ACKNOWLEDGED = "acknowledged"
    LISTENING = "listening"
    FINALIZING = "finalizing"
    REVIEW = "review"
    ANSWERING = "answering"
    CLOSED = "closed"


@dataclass(frozen=True)
class TurnConfig:
    segment_boundary_ms: int = 800
    final_silence_ms: int = 2500
    mute_drain_ms: int = 300
    first_speech_timeout_ms: int = 30000
    max_duration_ms: int = 45000

    @classmethod
    def from_env(cls) -> "TurnConfig":
        return cls(
            segment_boundary_ms=_bounded("QUESTION_SEGMENT_MS", 800, 200, 1500),
            final_silence_ms=_bounded("QUESTION_FINAL_SILENCE_MS", 2500, 1500, 6000),
            mute_drain_ms=_bounded("QUESTION_MUTE_DRAIN_MS", 300, 100, 1000),
            first_speech_timeout_ms=_bounded("QUESTION_FIRST_SPEECH_MS", 30000, 5000, 60000),
            max_duration_ms=_bounded("QUESTION_MAX_DURATION_MS", 45000, 10000, 90000),
        )


@dataclass(frozen=True)
class TurnMetric:
    turn_id: str
    state: TurnState
    elapsed_ms: int
    reason_code: str | None = None


@dataclass
class _Turn:
    turn_id: str
    started_ms: int
    state: TurnState = TurnState.IDLE
    first_speech_ms: int | None = None
    last_speech_ms: int | None = None
    mute_requested_ms: int | None = None
    tasks: list[asyncio.Task[str]] = field(default_factory=list)
    confirmed: bool = False


class QuestionTurnController:
    def __init__(self, *, config: TurnConfig | None = None, clock: Callable[[], float] = time.monotonic, id_factory: Callable[[], str] = lambda: uuid.uuid4().hex) -> None:
        self.config, self.clock, self.id_factory = config or TurnConfig.from_env(), clock, id_factory
        self.turn: _Turn | None = None
        self.metrics: list[TurnMetric] = []
        self.protocol_violations: list[str] = []
        self.review_ready = asyncio.Event()
        self.transcript: str | None = None

    @property
    def state(self) -> TurnState:
        return self.turn.state if self.turn else TurnState.IDLE

    @property
    def turn_id(self) -> str | None:
        return self.turn.turn_id if self.turn else None

    def start(self) -> str | None:
        if self.turn and self.turn.state is not TurnState.CLOSED:
            self._violation("duplicate_raise")
            return None
        self.review_ready = asyncio.Event()
        self.transcript = None
        self.turn = _Turn(self.id_factory(), self._now_ms())
        self.metrics.append(TurnMetric(self.turn.turn_id, TurnState.IDLE, 0))
        self._transition(TurnState.ACKNOWLEDGED)
        return self.turn.turn_id

    def listen(self) -> bool:
        if self.state is not TurnState.ACKNOWLEDGED:
            self._violation("listen_out_of_state")
            return False
        self._transition(TurnState.LISTENING)
        return True

    def observe_speech(self) -> None:
        if self.state is not TurnState.LISTENING:
            return
        now = self._now_ms()
        if self.turn.first_speech_ms is None:
            self.turn.first_speech_ms = now
        self.turn.last_speech_ms = now

    def add_stt(self, turn_id: str, result: Awaitable[str]) -> bool:
        if self.state is not TurnState.LISTENING or turn_id != self.turn_id:
            if asyncio.iscoroutine(result):
                result.close()
            self._violation("stale_result")
            return False
        self.turn.tasks.append(asyncio.create_task(result))
        return True

    def request_mute(self) -> None:
        if self.state is TurnState.LISTENING and self.turn.first_speech_ms is not None:
            self.turn.mute_requested_ms = self._now_ms()

    def endpoint_reason(self) -> str | None:
        if self.state is not TurnState.LISTENING:
            return None
        now = self._now_ms()
        if self.turn.first_speech_ms is None:
            return "no_speech" if now - self.turn.started_ms >= self.config.first_speech_timeout_ms else None
        if now - self.turn.first_speech_ms >= self.config.max_duration_ms:
            return "max_duration"
        if self.turn.mute_requested_ms is not None and now - self.turn.mute_requested_ms >= self.config.mute_drain_ms:
            return "mic_muted"
        if self.turn.last_speech_ms is not None and now - self.turn.last_speech_ms >= self.config.final_silence_ms:
            return "final_silence"
        return None

    async def finalize(self, reason_code: str) -> str | None:
        if self.state is not TurnState.LISTENING:
            self._violation("finalize_out_of_state")
            return None
        if reason_code in {"no_speech", "max_duration"} and (reason_code == "no_speech" or not self.turn.tasks):
            await self.close(reason_code)
            return None
        self._transition(TurnState.FINALIZING, reason_code)
        turn_id = self.turn.turn_id
        results = await asyncio.gather(*self.turn.tasks, return_exceptions=True)
        if self.turn is None or self.turn.turn_id != turn_id or self.state is TurnState.CLOSED:
            self._violation("stale_result")
            return None
        fragments = [value.strip() for value in results if isinstance(value, str) and value.strip()]
        transcript = _normalize(" ".join(fragments))
        self.turn.tasks.clear()
        if not transcript:
            await self.close("stt_empty")
            return None
        self.transcript = transcript
        self._transition(TurnState.REVIEW, reason_code)
        self.review_ready.set()
        return transcript

    def confirm(self, text: str) -> str | None:
        if self.state is not TurnState.REVIEW or self.turn.confirmed:
            self._violation("confirmation_out_of_state")
            return None
        normalized = _normalize(text)
        if not normalized:
            self._violation("empty_confirmation")
            return None
        self.turn.confirmed = True
        self._transition(TurnState.ANSWERING)
        return normalized

    async def cancel(self) -> bool:
        if self.state is not TurnState.REVIEW:
            self._violation("cancel_out_of_state")
            return False
        await self.close("cancelled")
        return True

    async def close(self, reason_code: str = "completed") -> None:
        if not self.turn:
            return
        for task in self.turn.tasks:
            if not task.done():
                task.cancel()
        if self.turn.tasks:
            await asyncio.gather(*self.turn.tasks, return_exceptions=True)
        self.turn.tasks.clear()
        self.transcript = None
        if self.state is not TurnState.CLOSED:
            self._transition(TurnState.CLOSED, reason_code)
        self.review_ready.set()

    def _transition(self, state: TurnState, reason_code: str | None = None) -> None:
        self.turn.state = state
        self.metrics.append(TurnMetric(self.turn.turn_id, state, self._now_ms() - self.turn.started_ms, reason_code))

    def _violation(self, reason: str) -> None:
        self.protocol_violations.append(reason)

    def _now_ms(self) -> int:
        return round(self.clock() * 1000)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _bounded(name: str, default: int, minimum: int, maximum: int) -> int:
    value = int(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum} ms")
    return value
