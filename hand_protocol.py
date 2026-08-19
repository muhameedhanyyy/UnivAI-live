"""Small, transport-independent helpers for the raised-hand data protocol."""

from __future__ import annotations

import uuid
from typing import Literal


HandState = Literal["raised", "acked", "lowered", "rejected"]


def normalize_hand_request_id(value: object) -> str | None:
    """Return a valid UUIDv4 request id while keeping legacy clients optional."""

    if not isinstance(value, str):
        return None
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        return None
    if parsed.version != 4:
        return None
    return value


def hand_event(
    state: HandState,
    request_id: str | None,
    *,
    detail: str | None = None,
) -> dict[str, str]:
    """Build one correlated browser event, with legacy no-id compatibility."""

    event = {"type": "hand", "state": state}
    if request_id is not None:
        event["request_id"] = request_id
    if detail is not None:
        event["detail"] = detail
    return event
