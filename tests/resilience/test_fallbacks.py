from resilience.fallbacks import FallbackMode, choose_fallback


def test_each_failure_has_visible_reason_and_preserves_lecture():
    for stage, mode in (("stt", FallbackMode.TEXT_QUESTION), ("tts", FallbackMode.TEXT_ANSWER), ("agent", FallbackMode.DEFERRED_ANSWER)):
        decision = choose_fallback(stage, "timeout")
        assert decision.mode is mode
        assert decision.reason_code == "timeout"
        assert decision.learner_message and decision.preserves_lecture
