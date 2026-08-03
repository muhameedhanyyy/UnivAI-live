"""Deterministic startup trace and lightweight artifact index."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


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

    def remaining(self, limit_s: float = 8.0) -> float:
        value = limit_s - (self.clock() - self.started)
        if value <= 0:
            raise TimeoutError("startup exceeded its explicit failure budget")
        return value

    def payload(self, *, mode: str, outcome: str = "ready", reason_code: str | None = None) -> dict:
        return {
            "schema": "univai.live.startup-trace", "version": "1.0.0",
            "trace_id": self.trace_id, "mode": mode, "outcome": outcome,
            "reason_code": reason_code,
            "stages": [{"stage": mark.stage.value, "offset_ms": mark.offset_ms} for mark in self.marks],
            "total_ms": self.marks[-1].offset_ms if self.marks else 0,
        }


class ArtifactIndex:
    def __init__(self, lectures_root: Path) -> None:
        self.entries: dict[tuple[str, int], Path] = {}
        for script in lectures_root.glob("*/week-*/script.json"):
            try:
                sid, week = script.parent.parent.name, int(script.parent.name.removeprefix("week-"))
                meta = script.parent / "audio" / "meta.json"
                json.loads(script.read_text("utf-8"))
                metadata = json.loads(meta.read_text("utf-8"))
                if int(metadata["sample_rate"]) >= 8000 and (script.parent / "audio" / "s0-t0.npy").is_file():
                    self.entries[(sid, week)] = script.parent
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue

    def require(self, learner_id: str, week: int) -> Path:
        try:
            return self.entries[(learner_id, week)]
        except KeyError as exc:
            raise FileNotFoundError("pre-rendered lecture artifact is missing or invalid") from exc


class LazyDependencies:
    """STT/TTS are imported and loaded only after first playback is available."""
    def __init__(self) -> None:
        self._tts = None
        self._stt = None
        self._tts_lock = asyncio.Lock()
        self._stt_lock = asyncio.Lock()

    async def tts(self):
        async with self._tts_lock:
            if self._tts is None:
                from tts import load_live_engine
                self._tts = await asyncio.to_thread(load_live_engine)
        return self._tts

    async def stt(self):
        async with self._stt_lock:
            if self._stt is None:
                from common.device import whisper_settings
                from faster_whisper import WhisperModel
                import os
                device, compute_type = whisper_settings()
                size = os.getenv("STT_MODEL_SIZE", "base")
                self._stt = await asyncio.to_thread(WhisperModel, size, device=device, compute_type=compute_type)
        return self._stt

    async def warm(self) -> None:
        await asyncio.gather(self.tts(), self.stt(), return_exceptions=True)
