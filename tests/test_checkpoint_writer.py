import asyncio
import threading

from checkpoint_writer import CheckpointWriter


def test_checkpoint_writes_do_not_block_narration_and_coalesce_updates():
    async def scenario() -> list[int]:
        writes: list[int] = []
        release_first = threading.Event()

        def write(checkpoint: int) -> None:
            writes.append(checkpoint)
            if len(writes) == 1:
                release_first.wait(timeout=2)

        writer = CheckpointWriter(write)
        writer.record(1)
        await asyncio.sleep(0.02)
        writer.record(2)
        writer.record(3)

        assert writes == [1]
        release_first.set()
        await writer.flush()
        return writes

    assert asyncio.run(scenario()) == [1, 3]


def test_checkpoint_writer_never_moves_backwards():
    async def scenario() -> list[int]:
        writes: list[int] = []
        writer = CheckpointWriter(writes.append, initial=4)
        writer.record(3)
        writer.record(5)
        await writer.flush()
        return writes

    assert asyncio.run(scenario()) == [5]
