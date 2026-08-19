from pathlib import Path
import io

from prompt_cache import PromptCache


class Audio(list):
    dtype = "float32"
    ndim = 1
    @property
    def size(self): return len(self)


class FakeNumpy:
    float32 = "float32"
    @staticmethod
    def array(values, dtype=None): return Audio(values)
    @staticmethod
    def ones(size, dtype=None): return Audio([1.0] * size)
    @staticmethod
    def zeros(size, dtype=None): return Audio([0.0] * size)
    @staticmethod
    def asarray(values, dtype=None): return Audio(values)
    @staticmethod
    def isfinite(values):
        class Result:
            @staticmethod
            def all(): return True
        return Result()
    @staticmethod
    def save(stream, audio, allow_pickle=False):
        stream.write(("audio:" + ",".join(map(str, audio))).encode())
    @staticmethod
    def load(source, allow_pickle=False):
        payload = source.read() if hasattr(source, "read") else Path(source).read_bytes()
        if not payload.startswith(b"audio:"):
            raise ValueError("invalid audio")
        values = payload.removeprefix(b"audio:").decode()
        return Audio(float(item) for item in values.split(",") if item)


def cache_with_fake_numpy(tmp_path, monkeypatch, **kwargs):
    import prompt_cache
    monkeypatch.setattr(prompt_cache, "_numpy", lambda: FakeNumpy)
    return PromptCache(tmp_path, **kwargs)


def test_prewarm_is_idempotent_and_tenant_isolated(tmp_path: Path, monkeypatch):
    renders = 0
    def render(text):
        nonlocal renders
        renders += 1
        return FakeNumpy.array([len(text), 1], dtype=FakeNumpy.float32)
    cache = cache_with_fake_numpy(tmp_path, monkeypatch)
    args = dict(language="en", voice="heart", model="kokoro", model_version="1", sample_rate=24000, render=render)
    first = cache.prewarm(learner_id="opaque-m", display_name="Mohamed Hany", **args)
    cache.prewarm(learner_id="opaque-m", display_name="Mohamed Hany", **args)
    cache.prewarm(learner_id="opaque-s", display_name="Sara Ali", **args)
    assert renders == 10
    assert first.normalized_name_digest not in str(tmp_path)
    assert "Mohamed" not in str(list(tmp_path.rglob("*")))


def test_corrupt_clip_uses_generic_and_queues_repair(tmp_path: Path, monkeypatch):
    repairs = []
    cache = cache_with_fake_numpy(tmp_path, monkeypatch, repair=repairs.append)
    args = dict(learner_id="opaque-m", display_name="Mohamed Hany", language="en", voice="heart", model="kokoro", model_version="1")
    manifest = cache.prewarm(**args, sample_rate=24000, render=lambda _: FakeNumpy.ones(2, dtype=FakeNumpy.float32))
    clip = next(tmp_path.rglob(manifest.clips[0].filename))
    clip.write_bytes(b"corrupt")
    generic = {"ask": FakeNumpy.zeros(1, dtype=FakeNumpy.float32)}
    loaded, rate = cache.load(**args, generic=generic)
    assert loaded is generic and rate is None
    assert repairs == ["personalized_prompt_cache_miss"]
