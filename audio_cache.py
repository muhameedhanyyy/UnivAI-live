"""Disposable, cross-process PCM cache for database-owned lecture narration."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path

import numpy as np

_SAFE_DIGEST = re.compile(r"^[a-f0-9]{64}$")


def script_digest(script: dict) -> str:
    """Digest exactly what the voice will say, and nothing else.

    Hashing the whole payload folded in fields that never reach the speaker —
    above all ``lectureId``, which differs per learner — so two copies of one
    lecture keyed apart and every sentence was rendered twice. The narration
    text in order is the complete determinant of the audio.
    """
    segments = script.get("segments") if isinstance(script, dict) else None
    spoken = [
        str(segment.get("text") or "")
        for segment in (segments or [])
        if isinstance(segment, dict)
    ]
    return hashlib.sha256(
        json.dumps(spoken, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class AudioCache:
    """Cache rendered sentences without making filesystem data canonical.

    The key is the digest of the script alone, never the artifact that carries
    it. Two learners who adopted the same course hold different artifact ids
    over identical narration, and keying on the artifact made each of them
    render every sentence again. A changed script still selects a new namespace
    because its digest changes. Files contain the sample rate with the PCM so a
    later worker may safely use another engine.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        configured = root or os.getenv("LIVE_AUDIO_CACHE_DIR", ".audio-cache")
        self.root = Path(configured).expanduser().resolve()

    def load(
        self,
        script_digest: str,
        segment: int,
        sentence: int,
    ) -> tuple[np.ndarray, int] | None:
        path = self._path(script_digest, segment, sentence)
        try:
            with np.load(path, allow_pickle=False) as saved:
                audio = np.asarray(saved["audio"], dtype=np.float32)
                sample_rate = int(saved["sample_rate"])
            if audio.ndim != 1 or not audio.size or sample_rate < 8000:
                return None
            return audio, sample_rate
        except (OSError, KeyError, TypeError, ValueError):
            return None

    def store(
        self,
        script_digest: str,
        segment: int,
        sentence: int,
        audio: np.ndarray,
        sample_rate: int,
    ) -> None:
        if not len(audio) or sample_rate < 8000:
            return
        target = self._path(script_digest, segment, sentence)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp.npz")
        np.savez_compressed(
            temporary,
            audio=np.asarray(audio, dtype=np.float32),
            sample_rate=np.asarray(sample_rate, dtype=np.int32),
        )
        os.replace(temporary, target)

    def _path(self, script_digest: str, segment: int, sentence: int) -> Path:
        # The digest is the whole key, so it is the whole trust boundary: it has
        # to be a hex SHA-256 and nothing else before it names a directory.
        if not _SAFE_DIGEST.fullmatch(script_digest):
            raise ValueError("audio cache script digest is unsafe")
        if segment < 0 or sentence < 0:
            raise ValueError("audio cache positions must be non-negative")
        return self.root / script_digest[:2] / script_digest / f"s{segment}-t{sentence}.npz"


__all__ = ["AudioCache", "script_digest"]
