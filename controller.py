"""Provider-free lecture controller used by the deterministic simulator."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from protocol import validate_inbound, validate_outbound, validate_script

Publish = Callable[[dict], Awaitable[None]]
Answer = Callable[[str], Awaitable[dict]]


class LectureController:
    def __init__(self, script: dict, publish: Publish, answer: Answer) -> None:
        validate_script(script)
        self.script = script
        self.publish = publish
        self.answer = answer
        self.cancelled = False
        self.muted = True
        self.hand_raised = False
        self.pending_question: str | None = None

    async def send(self, message: dict) -> None:
        validate_outbound(message)
        await self.publish(message)

    async def receive(self, message: dict) -> None:
        validate_inbound(message)
        kind = message["type"]
        if kind == "cancel":
            self.cancelled = True
        elif kind == "mic":
            self.muted = message["muted"]
        elif kind == "raise_hand":
            self.hand_raised = True
        elif kind == "question":
            self.pending_question = message["text"].strip()

    async def _handle_question(self) -> None:
        await self.send({"type": "state", "state": "asking"})
        await self.send({"type": "hand", "state": "acked"})
        if self.muted:
            await self.receive({"type": "mic", "muted": False})
        await self.send({"type": "state", "state": "listening"})
        await self.send({"type": "speech", "state": "waiting", "detail": "Speak now."})
        question = self.pending_question or "What protects each learner's material?"
        await self.send({"type": "speech", "state": "detected", "detail": "I can hear you."})
        await self.send({"type": "state", "state": "processing"})
        await self.send({"type": "speech", "state": "processing", "detail": "Turning your speech into text."})
        await self.send({"type": "state", "state": "review"})
        await self.send({"type": "transcript", "text": question})
        await self.send({"type": "speech", "state": "received", "detail": "Your transcript is ready."})
        await self.send(
            {"type": "progress", "stage": "retrieving", "detail": "fixture"}
        )
        await self.send({"type": "state", "state": "answering"})
        result = await self.answer(question)
        await self.send(
            {
                "type": "answer",
                "question": question,
                "answer": result["answer"],
                "pages": result.get("pages", []),
            }
        )
        await self.send({"type": "hand", "state": "lowered"})
        await self.send({"type": "transcript", "text": None})
        self.hand_raised = False
        self.pending_question = None

    async def run(self, *, delay: float = 0.0) -> None:
        await self.send({"type": "state", "state": "connecting"})
        await self.send({"type": "state", "state": "preparing"})
        for index, segment in enumerate(self.script["segments"]):
            if self.cancelled:
                break
            await self.send({"type": "slide", "n": segment["slide"]})
            await self.send({"type": "state", "state": "lecturing"})
            await self.send(
                {
                    "type": "progress",
                    "stage": "segment",
                    "detail": f"{index + 1}/{len(self.script['segments'])}",
                }
            )
            if delay:
                await asyncio.sleep(delay)
            if self.hand_raised and index == 0:
                await self._handle_question()
        await self.send({"type": "state", "state": "ended"})
