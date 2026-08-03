"""Deterministic, resumable playback of an already-grounded SectionPackV1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable

from protocols.section_session import SectionSessionMetaV1


class SectionState(str, Enum):
    INTRO = "intro"
    EXAMPLE = "example"
    GUIDED_TASK = "guided_task"
    WAITING = "waiting"
    FEEDBACK = "feedback"
    TODO_RECAP = "todo_recap"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


@dataclass(frozen=True)
class Checkpoint:
    activity_index: int = 0
    step_index: int = 0
    state: SectionState = SectionState.INTRO


class SectionController:
    def __init__(self, meta: SectionSessionMetaV1, emit: Callable[[dict], Awaitable[None]], speak: Callable[[str], Awaitable[None]]) -> None:
        self.meta, self.emit, self.speak = meta, emit, speak
        self.checkpoint = Checkpoint()
        self.todo_acknowledged: set[int] = set()
        self.submission_ids: set[str] = set()
        self.completed = False

    async def start(self) -> None:
        await self._publish(SectionState.INTRO, title=self.meta.pack["title"])
        await self.speak(self.meta.pack["title"])
        examples = self.meta.pack.get("examples", [])
        for activity_index, example in enumerate(examples[self.checkpoint.activity_index :], start=self.checkpoint.activity_index):
            await self._publish(SectionState.EXAMPLE, activity_index=activity_index, content=example)
            await self.speak(example["prompt"])
            for step_index, step in enumerate(example["steps"]):
                self.checkpoint = Checkpoint(activity_index, step_index, SectionState.EXAMPLE)
                await self.emit(self._event("section_step", content=step, activity_index=activity_index, step_index=step_index))
                await self.speak(f"{step['step']} {step['explanation']}")
        for activity_index, activity in enumerate(self.meta.pack["activities"]):
            self.checkpoint = Checkpoint(activity_index, 0, SectionState.GUIDED_TASK)
            await self._publish(SectionState.GUIDED_TASK, activity_index=activity_index, content=activity)
            await self.speak(activity["description"])
            await self._publish(SectionState.WAITING, activity_index=activity_index)
            break

    async def submit(self, submission_id: str, *, activity_index: int, text: str) -> bool:
        if not submission_id or submission_id in self.submission_ids:
            return False
        self.submission_ids.add(submission_id)
        await self._publish(SectionState.FEEDBACK, activity_index=activity_index, submission_id=submission_id, received=bool(text.strip()))
        await self._publish(SectionState.TODO_RECAP, todos=self.meta.pack.get("todos", []))
        return True

    async def acknowledge_todo(self, index: int) -> None:
        if index not in range(len(self.meta.pack.get("todos", []))):
            raise ValueError("unknown TODO index")
        self.todo_acknowledged.add(index)
        await self.emit(self._event("todo_acknowledged", todo_index=index))

    async def interrupt(self) -> Checkpoint:
        prior = self.checkpoint
        await self._publish(SectionState.INTERRUPTED, resume={"activity_index": prior.activity_index, "step_index": prior.step_index, "state": prior.state.value})
        self.checkpoint = prior
        return prior

    async def resume(self, checkpoint: Checkpoint) -> None:
        self.checkpoint = checkpoint
        await self.emit(self._event("section_resumed", activity_index=checkpoint.activity_index, step_index=checkpoint.step_index, state=checkpoint.state.value))

    async def complete(self) -> bool:
        if self.completed:
            return False
        self.completed = True
        await self._publish(SectionState.COMPLETED, todo_acknowledged=sorted(self.todo_acknowledged), attendance_changed=False)
        return True

    async def refuse_follow_up(self) -> None:
        await self.emit(self._event("grounded_refusal", message="That follow-up is not supported by the supplied section sources."))

    async def _publish(self, state: SectionState, **payload) -> None:
        self.checkpoint = Checkpoint(self.checkpoint.activity_index, self.checkpoint.step_index, state)
        await self.emit(self._event("section_state", state=state.value, **payload))

    def _event(self, event_type: str, **payload) -> dict:
        return {
            "type": event_type,
            "schema_version": "1.0.0",
            "session": {
                "learner_id": self.meta.learner_id,
                "programme_id": self.meta.programme_id,
                "course_id": self.meta.course_id,
                "week": self.meta.week,
                "lecture_id": self.meta.lecture_id,
                "plan_version": self.meta.plan_version,
            },
            "payload": payload,
        }
