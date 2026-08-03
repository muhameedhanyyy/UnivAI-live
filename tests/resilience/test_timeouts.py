import asyncio
import pytest

from resilience.timeouts import LatencyBudgets, Stage, StageTimeout, retry_bounded, within_budget


def test_stage_timeout_is_typed_and_bounded():
    async def slow():
        await asyncio.sleep(0.03)
    with pytest.raises(StageTimeout) as caught:
        asyncio.run(within_budget(Stage.STT, slow(), budgets=LatencyBudgets(stt=.01)))
    assert caught.value.stage is Stage.STT


def test_retry_is_bounded():
    calls = 0
    async def operation():
        nonlocal calls
        calls += 1
        if calls < 2:
            raise ConnectionError("temporary")
        return "ok"
    assert asyncio.run(retry_bounded(operation, base_backoff_s=0)) == "ok"
    assert calls == 2
