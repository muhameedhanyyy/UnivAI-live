"""Deterministic startup trace and lightweight artifact index."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from campus_imports import configure_campus_imports

configure_campus_imports()


def _db_fetch_all(sql: str) -> list[dict]:
    from common.db import fetch_all
    return fetch_all(sql)


def _db_fetch_one(sql: str, params: tuple[object, ...]) -> dict | None:
    from common.db import fetch_one
    return fetch_one(sql, params)


class StartupStage(str, Enum):
    DISPATCH = "dispatch"
    ROOM_CONNECTED = "room_connected"
    METADATA_VALID = "metadata_valid"
    ARTIFACT_LOADED = "artifact_loaded"
    TRACK_PUBLISHED = "track_published"
    READY_ACKNOWLEDGED = "ready_acknowledged"
    FIRST_FRAME = "first_frame"


ORDER = tuple(StartupStage)


@dataclass(frozen=True)
class StartupMark:
    stage: StartupStage
    offset_ms: int


class StartupTrace:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic, trace_id: str | None = None) -> None:
        self.clock, self.started = clock, clock()
        self.trace_id = trace_id or uuid.uuid4().hex
        self.marks: list[StartupMark] = []
        self.mark(StartupStage.DISPATCH)

    def mark(self, stage: StartupStage) -> StartupMark:
        if stage in {mark.stage for mark in self.marks}:
            raise ValueError(f"startup stage {stage.value} was already recorded")
        expected = ORDER[len(self.marks)]
        if stage is not expected:
            raise ValueError(f"startup stage {stage.value} is out of order; expected {expected.value}")
        mark = StartupMark(stage, round((self.clock() - self.started) * 1000))
        self.marks.append(mark)
        return mark

    def remaining(self, limit_s: float = 800.0) -> float:
        value = limit_s - (self.clock() - self.started)
        if value <= 0:
            raise TimeoutError("startup exceeded its explicit failure budget")
        return value

    def payload(self, *, mode: str, outcome: str = "ready", reason_code: str | None = None) -> dict:
        total_ms = (
            self.marks[-1].offset_ms
            if outcome == "ready" and self.marks
            else round((self.clock() - self.started) * 1000)
        )
        return {
            "schema": "univai.live.startup-trace", "version": "1.0.0",
            "trace_id": self.trace_id, "mode": mode, "outcome": outcome,
            "reason_code": reason_code,
            "stages": [{"stage": mark.stage.value, "offset_ms": mark.offset_ms} for mark in self.marks],
            "total_ms": total_ms,
        }


class ArtifactIndex:
    def __init__(self, _legacy_root: Path | None = None) -> None:
        # Do not scan every learner artifact in every idle job process. Entries
        # are verified lazily and then cached inside that process.
        self.entries: set[tuple[str, int]] = set()

    def require(self, learner_id: str, week: int) -> tuple[str, int]:
        key = (learner_id, week)
        if key not in self.entries:
            row = _db_fetch_one(
                """SELECT 1 FROM lecture_artifacts
                    WHERE student_id = %s AND week = %s
                      AND jsonb_array_length(script_payload->'segments') > 0
                    LIMIT 1""",
                (learner_id, week),
            )
            if not row:
                raise FileNotFoundError("database lecture artifact is missing or invalid")
            self.entries.add(key)
        return key


class LazyDependencies:
    """STT/TTS are imported and loaded only after first playback is available."""
    def __init__(self, *, tts=None) -> None:
        # Seed the engine loaded by AgentServer.setup_fnc. Without this, the
        # post-first-frame warm task loads the same model a second time.
        self._tts = tts
        self._stt = None
        self._stt_task: asyncio.Task | None = None
        self._tts_lock = asyncio.Lock()
        self._stt_lock = asyncio.Lock()

    async def tts(self):
        async with self._tts_lock:
            if self._tts is None:
                from tts import load_live_engine
                self._tts = await asyncio.to_thread(load_live_engine)
        return self._tts

    async def stt(self):
        # Keep one shared model load alive even if a learner-facing timeout
        # cancels its waiter.  Cancelling asyncio.to_thread cannot stop the
        # underlying model load; without this task cache, every retry started a
        # second Whisper model and could exhaust RAM.
        async with self._stt_lock:
            if self._stt is not None:
                return self._stt
            if self._stt_task is None:
                self._stt_task = asyncio.create_task(
                    asyncio.to_thread(self._load_stt),
                    name="load-whisper-model",
                )
            task = self._stt_task

        try:
            model = await asyncio.shield(task)
        except BaseException:
            async with self._stt_lock:
                if self._stt_task is task and task.done():
                    self._stt_task = None
            raise

        async with self._stt_lock:
            self._stt = model
            if self._stt_task is task:
                self._stt_task = None
        return self._stt

    @staticmethod
    def _load_stt():
        from common.device import whisper_settings
        from faster_whisper import WhisperModel
        import os

        device, compute_type = whisper_settings()
        size = os.getenv("STT_MODEL_SIZE", "base")
        return WhisperModel(size, device=device, compute_type=compute_type)

    async def warm(self) -> list:
        return await asyncio.gather(self.tts(), self.stt(), return_exceptions=True)
