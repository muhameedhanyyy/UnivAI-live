"""Bounded execution policies for the learner-facing Q&A pipeline."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class Stage(str, Enum):
    STT = "stt"
    RETRIEVAL_GENERATION = "retrieval_generation"
    TTS = "tts"
    TOTAL = "total"


class StageTimeout(TimeoutError):
    def __init__(self, stage: Stage, seconds: float) -> None:
        super().__init__(f"{stage.value} exceeded its {seconds:.1f}s budget")
        self.stage, self.seconds = stage, seconds


@dataclass(frozen=True)
class LatencyBudgets:
    stt: float = 8.0
    retrieval_generation: float = 15.0
    tts: float = 5.0
    total: float = 30.0

    @classmethod
    def from_env(cls) -> "LatencyBudgets":
        values = {
            "stt": _bounded("LIVE_STT_BUDGET_S", 8.0),
            "retrieval_generation": _bounded("LIVE_QA_BUDGET_S", 15.0),
            "tts": _bounded("LIVE_TTS_BUDGET_S", 5.0),
            "total": _bounded("LIVE_TOTAL_QA_BUDGET_S", 30.0),
        }
        result = cls(**values)
        if result.total < max(result.stt, result.retrieval_generation, result.tts):
            raise ValueError("total Q&A budget must cover every individual stage budget")
        return result

    def for_stage(self, stage: Stage) -> float:
        return float(getattr(self, stage.value))


async def within_budget(stage: Stage, operation: Awaitable[T], *, budgets: LatencyBudgets | None = None) -> T:
    seconds = (budgets or LatencyBudgets.from_env()).for_stage(stage)
    try:
        return await asyncio.wait_for(operation, timeout=seconds)
    except asyncio.TimeoutError as exc:
        raise StageTimeout(stage, seconds) from exc


async def retry_bounded(
    factory: Callable[[], Awaitable[T]],
    *,
    attempts: int = 2,
    base_backoff_s: float = 0.1,
    retryable: tuple[type[Exception], ...] = (TimeoutError, ConnectionError),
) -> T:
    if attempts not in range(1, 4):
        raise ValueError("attempts must be between 1 and 3")
    for attempt in range(attempts):
        try:
            return await factory()
        except retryable:
            if attempt + 1 == attempts:
                raise
            await asyncio.sleep(base_backoff_s * (2**attempt))
    raise AssertionError("unreachable")


def _bounded(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not 0.1 <= value <= 120:
        raise ValueError(f"{name} must be between 0.1 and 120 seconds")
    return value
