import numpy as np
import pytest

from audio_cache import AudioCache, script_digest


def test_audio_cache_round_trip_and_script_invalidation(tmp_path):
    cache = AudioCache(tmp_path)
    first = script_digest({"segments": [{"text": "one"}]})
    changed = script_digest({"segments": [{"text": "two"}]})
    audio = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)

    cache.store(first, 0, 0, audio, 24000)
    loaded = cache.load(first, 0, 0)
    assert loaded is not None
    np.testing.assert_allclose(loaded[0], audio)
    assert loaded[1] == 24000
    assert cache.load(changed, 0, 0) is None


def test_identical_narration_is_rendered_once_for_every_learner(tmp_path):
    """Two adopted courses hold one lecture under two identities.

    Each learner's script carries their own lectureId. Folding that into the
    key — or into the digest — made the second learner re-render every sentence
    of narration that was already on disk.
    """
    cache = AudioCache(tmp_path)
    script = {
        "title": "Transactions",
        "lectureId": "11111111-1111-4111-8111-111111111111",
        "segments": [{"slide": 1, "text": "A transaction is atomic."}],
    }
    adopted = dict(script, lectureId="22222222-2222-4222-8222-222222222222")
    audio = np.asarray([0.4, 0.5], dtype=np.float32)

    cache.store(script_digest(script), 0, 0, audio, 24000)

    hit = cache.load(script_digest(adopted), 0, 0)
    assert hit is not None
    np.testing.assert_allclose(hit[0], audio)


def test_the_digest_covers_the_narration_and_only_the_narration(tmp_path):
    spoken = {"segments": [{"text": "One."}, {"text": "Two."}]}
    # Identity and presentation never reach the speaker.
    assert script_digest(dict(spoken, lectureId="x", title="A")) == script_digest(
        dict(spoken, lectureId="y", title="B")
    )
    # Changed words, changed order, and dropped sentences all must invalidate.
    assert script_digest(spoken) != script_digest({"segments": [{"text": "One."}]})
    assert script_digest(spoken) != script_digest(
        {"segments": [{"text": "Two."}, {"text": "One."}]}
    )
    assert script_digest(spoken) != script_digest({"segments": [{"text": "One!"}, {"text": "Two."}]})
    # A malformed script must still produce a key rather than raise.
    assert len(script_digest({})) == 64


def test_audio_cache_rejects_a_digest_that_is_not_a_digest(tmp_path):
    cache = AudioCache(tmp_path)
    for unsafe in ("../escape", "digest", "", "a" * 63, "A" * 64):
        with pytest.raises(ValueError, match="unsafe"):
            cache.load(unsafe, 0, 0)
