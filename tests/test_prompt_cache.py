from pathlib import Path
import numpy as np

from prompt_cache import PromptCache


def test_prewarm_is_idempotent_and_tenant_isolated(tmp_path: Path):
    renders = 0
    def render(text):
        nonlocal renders
        renders += 1
        return np.array([len(text), 1], dtype=np.float32)
    cache = PromptCache(tmp_path)
    args = dict(language="en", voice="heart", model="kokoro", model_version="1", sample_rate=24000, render=render)
    first = cache.prewarm(learner_id="opaque-m", display_name="Mohamed Hany", **args)
    cache.prewarm(learner_id="opaque-m", display_name="Mohamed Hany", **args)
    cache.prewarm(learner_id="opaque-s", display_name="Sara Ali", **args)
    assert renders == 6
    assert first.normalized_name_digest not in str(tmp_path)
    assert "Mohamed" not in str(list(tmp_path.rglob("*")))


def test_corrupt_clip_uses_generic_and_queues_repair(tmp_path: Path):
    repairs = []
    cache = PromptCache(tmp_path, repair=repairs.append)
    args = dict(learner_id="opaque-m", display_name="Mohamed Hany", language="en", voice="heart", model="kokoro", model_version="1")
    manifest = cache.prewarm(**args, sample_rate=24000, render=lambda _: np.ones(2, dtype=np.float32))
    clip = next(tmp_path.rglob(manifest.clips[0].filename))
    clip.write_bytes(b"corrupt")
    generic = {"ask": np.zeros(1, dtype=np.float32)}
    loaded, rate = cache.load(**args, generic=generic)
    assert loaded is generic and rate is None
    assert repairs == ["personalized_prompt_cache_miss"]
