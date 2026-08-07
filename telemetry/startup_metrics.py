"""Raw startup traces and honest cold/warm summaries."""

from __future__ import annotations

import json
import math
from pathlib import Path


def append_trace(path: Path, trace: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(trace, sort_keys=True) + "\n")


def summarize(traces: list[dict]) -> dict:
    result = {}
    for mode in ("cold", "warm"):
        selected = [item for item in traces if item.get("mode") == mode]
        if len(selected) < 30:
            raise ValueError(f"{mode} startup summary requires at least 30 raw traces")
        successes = sorted(item["total_ms"] for item in selected if item.get("outcome") == "ready")
        if not successes:
            raise ValueError(f"{mode} startup summary has no successful traces")
        result[mode] = {
            "samples": len(selected), "failures": len(selected) - len(successes),
            "p50_ms": _percentile(successes, .50), "p95_ms": _percentile(successes, .95),
            "max_ms": max(successes),
        }
    return result


def summarize_audible(events: list[dict]) -> dict:
    """Summarize browser-confirmed first audio, not server frame capture."""
    result = {}
    for mode in ("cold", "warm"):
        selected = [
            item for item in events
            if item.get("schema") == "univai.live.client-first-audio"
            and item.get("mode") == mode
            and isinstance(item.get("client_elapsed_ms"), int)
        ]
        if len(selected) < 30:
            raise ValueError(f"{mode} audible startup summary requires at least 30 browser samples")
        values = sorted(item["client_elapsed_ms"] for item in selected)
        result[mode] = {
            "samples": len(values),
            "p50_ms": _percentile(values, .50),
            "p95_ms": _percentile(values, .95),
            "max_ms": max(values),
        }
    return result


def _percentile(values: list[int], percentile: float) -> int:
    return values[max(0, math.ceil(percentile * len(values)) - 1)]
