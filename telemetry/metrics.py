"""Structured latency telemetry; never accepts learner speech or answer text."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Callable


@dataclass(frozen=True)
class StageMetric:
    turn_id: str
    stage: str
    elapsed_ms: int
    outcome: str
    reason_code: str | None = None


class Metrics:
    def __init__(self, emit: Callable[[str], None] = print) -> None:
        self.emit = emit

    def record(self, metric: StageMetric) -> None:
        self.emit(json.dumps({"metric": "live_stage_latency", **asdict(metric)}, sort_keys=True))


class Timer:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock, self.started = clock, clock()

    def elapsed_ms(self) -> int:
        return round((self.clock() - self.started) * 1000)
