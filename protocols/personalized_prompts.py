"""PersonalizedPromptManifestV1 cache contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass

MANIFEST_SCHEMA = "univai.live.personalized-prompts"
MANIFEST_VERSION = "1.0.0"
PHRASE_SET_VERSION = "1.1.0"


@dataclass(frozen=True)
class ClipRecord:
    phrase_id: str
    filename: str
    sha256: str


@dataclass(frozen=True)
class PersonalizedPromptManifestV1:
    learner_id: str
    normalized_name_digest: str
    language: str
    voice: str
    model: str
    model_version: str
    sample_rate: int
    phrase_set_version: str
    clips: tuple[ClipRecord, ...]
    schema_name: str = MANIFEST_SCHEMA
    schema_version: str = MANIFEST_VERSION

    def as_dict(self) -> dict:
        value = asdict(self)
        value["clips"] = [asdict(clip) for clip in self.clips]
        return value

    @classmethod
    def from_dict(cls, value: dict) -> "PersonalizedPromptManifestV1":
        if value.get("schema_name") != MANIFEST_SCHEMA or value.get("schema_version") != MANIFEST_VERSION:
            raise ValueError("unsupported personalized prompt manifest")
        clips = tuple(ClipRecord(**clip) for clip in value.get("clips", []))
        result = cls(**{key: value[key] for key in (
            "learner_id", "normalized_name_digest", "language", "voice", "model",
            "model_version", "sample_rate", "phrase_set_version"
        )}, clips=clips)
        if not result.learner_id or result.sample_rate < 8000 or not clips:
            raise ValueError("invalid personalized prompt manifest")
        if len({clip.phrase_id for clip in clips}) != len(clips):
            raise ValueError("duplicate phrase IDs")
        return result
