"""Disposable, cross-process PCM cache for database-owned lecture narration."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path

import numpy as np

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def script_digest(script: dict) -> str:
    return hashlib.sha256(
        json.dumps(script, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class AudioCache:
    """Cache rendered sentences without making filesystem data canonical.

    The key is an opaque database artifact id plus a digest of its script. A
    changed script automatically selects a new namespace. Files contain the
    sample rate with the PCM so a later worker may safely use another engine.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        configured = root or os.getenv("LIVE_AUDIO_CACHE_DIR", ".audio-cache")
        self.root = Path(configured).expanduser().resolve()

    def load(
        self,
        artifact_id: str,
        script_digest: str,
        segment: int,
        sentence: int,
    ) -> tuple[np.ndarray, int] | None:
        path = self._path(artifact_id, script_digest, segment, sentence)
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
        artifact_id: str,
        script_digest: str,
        segment: int,
        sentence: int,
        audio: np.ndarray,
        sample_rate: int,
    ) -> None:
        if not len(audio) or sample_rate < 8000:
            return
        target = self._path(artifact_id, script_digest, segment, sentence)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp.npz")
        np.savez_compressed(
            temporary,
            audio=np.asarray(audio, dtype=np.float32),
            sample_rate=np.asarray(sample_rate, dtype=np.int32),
        )
        os.replace(temporary, target)

    def _path(self, artifact_id: str, script_digest: str, segment: int, sentence: int) -> Path:
        if not _SAFE_ID.fullmatch(artifact_id):
            raise ValueError("audio cache artifact id is unsafe")
        if segment < 0 or sentence < 0:
            raise ValueError("audio cache positions must be non-negative")
        digest = hashlib.sha256(script_digest.encode("utf-8")).hexdigest()[:20]
        return self.root / artifact_id / digest / f"s{segment}-t{sentence}.npz"


__all__ = ["AudioCache", "script_digest"]
