"""Fixed-template name personalization with injection-resistant normalization."""

from __future__ import annotations

import hashlib
import re
import unicodedata

PHRASES = {
    "ask": "Yes, {name}? Do you have a question? Unmute your microphone and go ahead.",
    "remind": "{name}, your hand is still raised. Unmute whenever you are ready; I am listening.",
    "resume": "No question? No problem. Let us continue with the lecture.",
    "rejoin": "Welcome back, {name}. I am continuing from three sentences before where we stopped.",
}


def normalize_display_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = " ".join(normalized.split())
    if not 1 <= len(normalized) <= 80:
        raise ValueError("display name must contain 1..80 characters")
    if any(not (char.isalpha() or char in " -'") for char in normalized):
        raise ValueError("display name contains unsupported speech characters")
    if not any(char.isalpha() for char in normalized):
        raise ValueError("display name must contain a letter")
    return normalized


def normalized_name_digest(value: str) -> str:
    normalized = normalize_display_name(value).casefold().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def render_templates(value: str) -> dict[str, str]:
    name = normalize_display_name(value)
    return {phrase_id: template.format(name=name) for phrase_id, template in PHRASES.items()}
