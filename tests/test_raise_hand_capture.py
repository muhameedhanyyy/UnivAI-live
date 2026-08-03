import asyncio

from question_turn import QuestionTurnController
from test_question_turn import Clock


def test_concurrent_sessions_and_turn_replacement_never_share_fragments():
    async def scenario():
        clock = Clock()
        left = QuestionTurnController(clock=clock, id_factory=lambda: "left")
        right = QuestionTurnController(clock=clock, id_factory=lambda: "right")
        for controller in (left, right): controller.start(); controller.listen(); controller.observe_speech()
        left.add_stt("left", asyncio.sleep(0, result="left question"))
        right.add_stt("right", asyncio.sleep(0, result="right question"))
        clock.advance(2.5)
        assert await left.finalize("final_silence") == "left question"
        assert await right.finalize("final_silence") == "right question"
    asyncio.run(scenario())


def test_one_word_confirmed_question_is_allowed_once():
    async def scenario():
        clock = Clock(); controller = QuestionTurnController(clock=clock, id_factory=lambda: "turn")
        controller.start(); controller.listen(); controller.observe_speech()
        controller.add_stt("turn", asyncio.sleep(0, result="Why")); clock.advance(2.5)
        await controller.finalize("final_silence")
        assert controller.confirm("Why") == "Why"
    asyncio.run(scenario())
