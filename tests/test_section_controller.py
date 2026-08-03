import asyncio, json

from protocols.section_session import SectionSessionMetaV1
from section_controller import SectionController, SectionState
from test_section_session import metadata


def test_playback_resume_idempotency_and_explicit_completion():
    async def scenario():
        events, spoken = [], []
        async def emit(event): events.append(event)
        async def speak(text): spoken.append(text)
        meta = SectionSessionMetaV1.from_room_metadata(json.dumps(metadata()), authenticated_learner_id="u1")
        controller = SectionController(meta, emit, speak)
        await controller.start()
        checkpoint = await controller.interrupt()
        assert checkpoint.state is SectionState.WAITING
        assert await controller.submit("s1", activity_index=0, text="answer") is True
        assert await controller.submit("s1", activity_index=0, text="duplicate") is False
        await controller.acknowledge_todo(0)
        assert await controller.complete() is True
        assert await controller.complete() is False
        assert events[-1]["payload"]["attendance_changed"] is False
        assert all(event["session"]["learner_id"] == "u1" for event in events)
        assert spoken
    asyncio.run(scenario())
