"""Small, SDK-agnostic helpers for finding an authenticated learner's mic."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def is_learner_microphone(
    track: Any,
    publication: Any,
    participant: Any,
    learner_id: str,
    *,
    audio_kind: Any,
    microphone_source: Any,
) -> bool:
    """Reject screen-share audio and tracks owned by another participant."""
    return bool(
        track is not None
        and publication is not None
        and participant is not None
        and getattr(participant, "identity", None) == learner_id
        and getattr(track, "kind", None) == audio_kind
        and getattr(publication, "source", None) == microphone_source
    )


def existing_learner_microphones(
    room: Any,
    learner_id: str,
    *,
    audio_kind: Any,
    microphone_source: Any,
) -> Iterator[tuple[Any, Any, Any]]:
    """Yield subscribed mics that existed before a handler was registered.

    LiveKit may subscribe the worker to a learner track immediately after
    ``connect()``. Registering ``track_subscribed`` later does not replay that
    event, so callers must also inspect current publications.
    """
    participants = getattr(room, "remote_participants", {})
    for participant in participants.values():
        if getattr(participant, "identity", None) != learner_id:
            continue
        publications = getattr(participant, "track_publications", {})
        for publication in publications.values():
            track = getattr(publication, "track", None)
            if is_learner_microphone(
                track,
                publication,
                participant,
                learner_id,
                audio_kind=audio_kind,
                microphone_source=microphone_source,
            ):
                yield track, publication, participant


def track_key(track: Any) -> str:
    """Use LiveKit's stable SID and retain an object fallback for test doubles."""
    return str(getattr(track, "sid", "") or id(track))
