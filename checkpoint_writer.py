"""Coalesced background persistence for live narration checkpoints."""

from __future__ import annotations

import asyncio
from collections.abc import Callable


class CheckpointWriter:
    """Persist the newest checkpoint without pausing sentence playback."""

    def __init__(self, write: Callable[[int], None], initial: int = 0) -> None:
        self._write = write
        self._latest = max(0, int(initial))
        self._persisted = self._latest
        self._event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._closing = False

    def record(self, checkpoint: int) -> None:
        if self._closing:
            raise RuntimeError("Checkpoint writer is closing")
        self._latest = max(self._latest, int(checkpoint))
        if self._latest <= self._persisted:
            return
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        self._event.set()

    async def flush(self) -> None:
        self._closing = True
        if self._task is None:
            return
        self._event.set()
        await self._task

    async def _run(self) -> None:
        while True:
            await self._event.wait()
            self._event.clear()
            target = self._latest
            if target > self._persisted:
                await asyncio.to_thread(self._write, target)
                self._persisted = target
            if self._latest > self._persisted:
                self._event.set()
            elif self._closing:
                return


__all__ = ["CheckpointWriter"]
