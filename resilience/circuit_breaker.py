"""Small monotonic circuit breaker with no provider-specific coupling."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpen(RuntimeError):
    pass


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    recovery_timeout_s: float = 20.0
    clock: Callable[[], float] = time.monotonic
    failures: int = 0
    opened_at: float | None = None
    _probe_active: bool = field(default=False, init=False)

    @property
    def state(self) -> CircuitState:
        if self.opened_at is None:
            return CircuitState.CLOSED
        if self.clock() - self.opened_at >= self.recovery_timeout_s:
            return CircuitState.HALF_OPEN
        return CircuitState.OPEN

    def before_call(self) -> None:
        state = self.state
        if state is CircuitState.OPEN or (state is CircuitState.HALF_OPEN and self._probe_active):
            raise CircuitOpen("provider circuit is open")
        if state is CircuitState.HALF_OPEN:
            self._probe_active = True

    def record_success(self) -> None:
        self.failures, self.opened_at, self._probe_active = 0, None, False

    def record_failure(self) -> None:
        self._probe_active = False
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = self.clock()
