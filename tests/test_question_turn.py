import asyncio
import pytest

from question_turn import QuestionTurnController, TurnConfig, TurnState


class Clock:
    now = 0.0
    def __call__(self): return self.now
    def advance(self, seconds): self.now += seconds


def test_natural_pauses_join_all_chunks_before_review_and_confirmation():
    async def scenario():
        clock = Clock(); controller = QuestionTurnController(clock=clock, id_factory=lambda: "A")
        assert controller.start() == "A"; assert controller.listen()
        controller.observe_speech(); controller.add_stt("A", asyncio.sleep(0, result="What is"))
        clock.advance(1); assert controller.endpoint_reason() is None
        controller.observe_speech(); controller.add_stt("A", asyncio.sleep(0, result="gradient descent"))
        clock.advance(2); assert controller.endpoint_reason() is None
        clock.advance(.5); assert controller.endpoint_reason() == "final_silence"
        assert await controller.finalize("final_silence") == "What is gradient descent"
        assert controller.state is TurnState.REVIEW
        assert controller.confirm("What is gradient descent") == "What is gradient descent"
        assert controller.confirm("duplicate") is None
        await controller.close("completed")
        assert [metric.state for metric in controller.metrics] == [
            TurnState.IDLE, TurnState.ACKNOWLEDGED, TurnState.LISTENING,
            TurnState.FINALIZING, TurnState.REVIEW, TurnState.ANSWERING, TurnState.CLOSED,
        ]
    asyncio.run(scenario())


def test_exact_final_silence_and_mute_drain_boundaries():
    async def scenario():
        clock = Clock(); controller = QuestionTurnController(clock=clock, id_factory=lambda: "A")
        controller.start(); controller.listen(); controller.observe_speech()
        controller.add_stt("A", asyncio.sleep(0, result="Why"))
        clock.advance(2.49); assert controller.endpoint_reason() is None
        clock.advance(.01); assert controller.endpoint_reason() == "final_silence"
        await controller.close("test")
        controller.start(); controller.listen(); controller.observe_speech()
        controller.add_stt("A", asyncio.sleep(0, result="How")); controller.request_mute()
        clock.advance(.299); assert controller.endpoint_reason() is None
        clock.advance(.001); assert controller.endpoint_reason() == "mic_muted"
    asyncio.run(scenario())


def test_late_cancel_timeout_and_protocol_messages_never_confirm():
    async def scenario():
        clock = Clock(); ids = iter(("A", "B")); controller = QuestionTurnController(clock=clock, id_factory=lambda: next(ids))
        controller.start(); controller.listen()
        assert controller.confirm("too early") is None
        await controller.close("cancelled")
        controller.start(); controller.listen()
        assert not controller.add_stt("A", asyncio.sleep(0, result="stale words"))
        clock.advance(30); assert controller.endpoint_reason() == "no_speech"
        assert await controller.finalize("no_speech") == ""
        assert controller.state is TurnState.REVIEW and controller.transcript == ""
        assert controller.review_reason == "no_speech"
    asyncio.run(scenario())


def test_invalid_environment_is_rejected(monkeypatch):
    monkeypatch.setenv("QUESTION_FINAL_SILENCE_MS", "800")
    with pytest.raises(ValueError, match="QUESTION_FINAL_SILENCE_MS"):
        TurnConfig.from_env()


def test_explicit_mute_never_waits_for_no_speech_timeout_and_empty_stt_can_be_typed():
    async def scenario():
        clock = Clock()
        controller = QuestionTurnController(clock=clock, id_factory=lambda: "A")
        controller.start(); controller.listen(); controller.request_mute()
        clock.advance(.3)
        assert controller.endpoint_reason() == "mic_muted"
        assert await controller.finalize("mic_muted") == ""
        assert controller.state is TurnState.REVIEW
        assert controller.transcript == ""
        assert controller.confirm("typed fallback question") == "typed fallback question"

    asyncio.run(scenario())


def test_processing_is_bounded_and_an_empty_turn_can_retry_or_cancel():
    async def never_finishes():
        await asyncio.Event().wait()
        return "unreachable"

    async def scenario():
        clock = Clock()
        controller = QuestionTurnController(
            config=TurnConfig(processing_timeout_ms=5),
            clock=clock,
            id_factory=iter(("A", "B")).__next__,
        )
        controller.start(); controller.listen(); controller.observe_speech()
        controller.add_stt("A", never_finishes())
        assert await controller.finalize("final_silence") == ""
        assert controller.state is TurnState.REVIEW
        assert controller.review_reason == "processing_timeout"

        assert await controller.retry() == "B"
        assert controller.state is TurnState.LISTENING
        assert controller.transcript is None
        assert controller.review_reason is None
        assert await controller.cancel()
        assert controller.state is TurnState.CLOSED

    asyncio.run(scenario())
