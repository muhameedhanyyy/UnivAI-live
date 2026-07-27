"""Explicit runtime selection for the Live repository."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path


class RuntimeMode(str, Enum):
    STANDALONE = "standalone"
    INTEGRATED = "integrated"


REPOSITORY_ROOT = Path(__file__).resolve().parent


def runtime_mode() -> RuntimeMode:
    raw = os.getenv("UNIVAI_MODE", RuntimeMode.INTEGRATED.value).strip().lower()
    try:
        mode = RuntimeMode(raw)
    except ValueError as exc:
        raise RuntimeError("UNIVAI_MODE must be standalone or integrated") from exc
    if mode is RuntimeMode.STANDALONE and os.getenv("UNIVAI_ENV", "").lower() in {
        "production",
        "prod",
    }:
        raise RuntimeError("Standalone Live providers are disabled in production")
    return mode
