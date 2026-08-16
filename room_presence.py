"""SDK-agnostic checks for authenticated learner presence in a LiveKit room."""

from __future__ import annotations

from typing import Any


def learner_is_in_room(room: Any, learner_id: str) -> bool:
    """Return whether LiveKit still lists the exact learner as a participant."""

    participants = getattr(room, "remote_participants", {})
    values = participants.values() if hasattr(participants, "values") else participants
    return any(
        getattr(participant, "identity", None) == learner_id
        for participant in values
    )
