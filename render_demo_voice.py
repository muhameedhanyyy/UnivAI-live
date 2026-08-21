"""Render one private demo interaction clip with the live Piper lecturer voice."""

from __future__ import annotations

import argparse
import base64
import json
import os

os.environ["TTS_LIVE_ENGINE"] = "piper"
from prepare_demo_media import Renderer, wav_info


def decode_text(value: str) -> str:
    padding = "=" * (-len(value) % 4)
    text = base64.urlsafe_b64decode(value + padding).decode("utf-8")
    normalized = " ".join(text.split())
    if not 1 <= len(normalized) <= 4_000:
        raise ValueError("voice text must contain 1..4000 characters")
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    args = parser.parse_args()
    clip = Renderer().clip(decode_text(args.text))
    _, _, duration_ms = wav_info(clip)
    print(json.dumps({"ok": True, "durationMs": duration_ms}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
