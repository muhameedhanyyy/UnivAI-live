import numpy as np

from audio_cache import AudioCache, script_digest


def test_audio_cache_round_trip_and_script_invalidation(tmp_path):
    cache = AudioCache(tmp_path)
    first = script_digest({"segments": [{"text": "one"}]})
    changed = script_digest({"segments": [{"text": "two"}]})
    audio = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)

    cache.store("artifact_1", first, 0, 0, audio, 24000)
    loaded = cache.load("artifact_1", first, 0, 0)
    assert loaded is not None
    np.testing.assert_allclose(loaded[0], audio)
    assert loaded[1] == 24000
    assert cache.load("artifact_1", changed, 0, 0) is None


def test_audio_cache_rejects_path_traversal(tmp_path):
    cache = AudioCache(tmp_path)
    try:
        cache.load("../escape", "digest", 0, 0)
    except ValueError as exc:
        assert "unsafe" in str(exc)
    else:
        raise AssertionError("unsafe artifact id was accepted")
