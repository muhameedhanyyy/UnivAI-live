"""Atomic, tenant-isolated personalized prompt audio cache."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Callable

from personalization import normalized_name_digest, render_templates
from protocols.personalized_prompts import ClipRecord, PHRASE_SET_VERSION, PersonalizedPromptManifestV1
from resilience.timeouts import Stage, within_budget


class PromptCache:
    def __init__(self, root: Path, *, repair: Callable[[str], None] | None = None) -> None:
        self.root = root
        self.repair = repair or (lambda _reason: None)

    def prewarm(
        self,
        *,
        learner_id: str,
        display_name: str,
        language: str,
        voice: str,
        model: str,
        model_version: str,
        sample_rate: int,
        render: Callable[[str], Any],
    ) -> PersonalizedPromptManifestV1:
        target = self._directory(learner_id, display_name, language, voice, model, model_version)
        existing = self._read_manifest(target)
        if existing and self._valid_clips(target, existing):
            return existing
        target.mkdir(parents=True, exist_ok=True)
        records: list[ClipRecord] = []
        for phrase_id, text in render_templates(display_name).items():
            np = _numpy()
            audio = np.asarray(render(text), dtype=np.float32)
            if not audio.size or not np.isfinite(audio).all():
                raise ValueError("renderer returned invalid audio")
            payload = _npy_bytes(audio)
            filename = f"{phrase_id}.npy"
            _atomic_write(target / filename, payload)
            records.append(ClipRecord(phrase_id, filename, hashlib.sha256(payload).hexdigest()))
        manifest = PersonalizedPromptManifestV1(
            learner_id=learner_id,
            normalized_name_digest=normalized_name_digest(display_name),
            language=language,
            voice=voice,
            model=model,
            model_version=model_version,
            sample_rate=sample_rate,
            phrase_set_version=PHRASE_SET_VERSION,
            clips=tuple(records),
        )
        _atomic_write(target / "manifest.json", json.dumps(manifest.as_dict(), sort_keys=True).encode())
        return manifest

    def load(
        self,
        *,
        learner_id: str,
        display_name: str,
        language: str,
        voice: str,
        model: str,
        model_version: str,
        generic: dict[str, Any],
    ) -> tuple[dict[str, Any], int | None]:
        target = self._directory(learner_id, display_name, language, voice, model, model_version)
        manifest = self._read_manifest(target)
        if manifest is None or not self._valid_clips(target, manifest):
            self.repair("personalized_prompt_cache_miss")
            return generic, None
        np = _numpy()
        clips = {clip.phrase_id: np.load(target / clip.filename, allow_pickle=False) for clip in manifest.clips}
        return clips, manifest.sample_rate

    def invalidate_learner(self, learner_id: str) -> None:
        tenant = self.root / _digest(learner_id)
        if not tenant.is_dir() or tenant.parent.resolve() != self.root.resolve():
            return
        for path in sorted(tenant.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        tenant.rmdir()

    def _directory(self, learner_id: str, display_name: str, language: str, voice: str, model: str, model_version: str) -> Path:
        if not learner_id or any(char in learner_id for char in "/\\"):
            raise ValueError("opaque learner ID is required")
        identity = "|".join((normalized_name_digest(display_name), language, voice, model, model_version, PHRASE_SET_VERSION))
        return self.root / _digest(learner_id) / _digest(identity)

    @staticmethod
    def _read_manifest(target: Path) -> PersonalizedPromptManifestV1 | None:
        try:
            return PersonalizedPromptManifestV1.from_dict(json.loads((target / "manifest.json").read_text("utf-8")))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _valid_clips(target: Path, manifest: PersonalizedPromptManifestV1) -> bool:
        np = _numpy()
        try:
            for clip in manifest.clips:
                if Path(clip.filename).name != clip.filename:
                    return False
                payload = (target / clip.filename).read_bytes()
                if hashlib.sha256(payload).hexdigest() != clip.sha256:
                    return False
                audio = np.load(io.BytesIO(payload), allow_pickle=False)
                if audio.dtype != np.float32 or audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
                    return False
            return True
        except (OSError, ValueError):
            return False


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _npy_bytes(audio: Any) -> bytes:
    np = _numpy()
    stream = io.BytesIO()
    np.save(stream, audio, allow_pickle=False)
    return stream.getvalue()


def _numpy():
    import numpy
    return numpy


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


async def prewarm_authenticated(cache: PromptCache, *, authenticated_learner_id: str, learner_id: str, **kwargs) -> PersonalizedPromptManifestV1:
    """Bounded registration/profile-change hook; authentication is fail-closed."""
    if not authenticated_learner_id or authenticated_learner_id != learner_id:
        raise PermissionError("authenticated learner does not own this prompt cache")
    return await within_budget(Stage.TOTAL, asyncio.to_thread(cache.prewarm, learner_id=learner_id, **kwargs))
