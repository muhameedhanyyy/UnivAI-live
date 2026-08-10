"""Local deterministic Live lecture simulator."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from controller import LectureController
from protocol import validate_outbound, validate_script
from runtime import REPOSITORY_ROOT, RuntimeMode, runtime_mode


async def fixture_answer(question: str) -> dict:
    lowered = question.lower()
    if "tenant" in lowered or "learner" in lowered or "material" in lowered:
        return {
            "answer": "Tenant filtering keeps each learner's material separate.",
            "pages": [2],
            "model_used": "standalone-fixture",
        }
    return {
        "answer": "That is not covered in your book, so I cannot answer it from the material.",
        "pages": [],
        "model_used": "standalone-fixture",
    }


async def run_simulation(*, trace: bool = True, cancel: bool = False) -> list[dict]:
    if runtime_mode() is not RuntimeMode.STANDALONE:
        raise RuntimeError("The simulator requires UNIVAI_MODE=standalone")
    script = json.loads(
        (REPOSITORY_ROOT / "fixtures" / "lecture.json").read_text(encoding="utf-8")
    )
    validate_script(script)
    messages: list[dict] = []

    async def publish(message: dict) -> None:
        validate_outbound(message)
        messages.append(message)
        if trace:
            print(json.dumps(message, ensure_ascii=False))

    controller = LectureController(script, publish, fixture_answer)
    await controller.receive({"type": "raise_hand"})
    await controller.receive({"type": "mic", "muted": False})
    await controller.receive(
        {"type": "question", "text": "What protects each learner's material?"}
    )
    if cancel:
        await controller.receive({"type": "cancel"})
    await controller.run()
    return messages


def smoke() -> int:
    messages = asyncio.run(run_simulation(trace=False))
    kinds = {message["type"] for message in messages}
    states = {
        message["state"] for message in messages if message["type"] == "state"
    }
    required_kinds = {"slide", "state", "progress", "transcript", "answer", "hand", "speech"}
    required_states = {
        "connecting",
        "preparing",
        "lecturing",
        "asking",
        "listening",
        "processing",
        "review",
        "answering",
        "ended",
    }
    if not required_kinds <= kinds or not required_states <= states:
        raise RuntimeError("standalone trace missed required protocol checkpoints")
    print(json.dumps({"ok": True, "messages": len(messages), "audio": "silent"}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("run", "smoke", "status"), default="run")
    parser.add_argument("--cancel", action="store_true")
    args = parser.parse_args()
    if args.command == "smoke":
        return smoke()
    if args.command == "status":
        print(
            json.dumps(
                {
                    "ok": True,
                    "mode": runtime_mode().value,
                    "fixture": str(REPOSITORY_ROOT / "fixtures" / "lecture.json"),
                    "tts": "silent fallback",
                    "stt": "scripted transcript",
                    "transport": "local trace",
                },
                indent=2,
            )
        )
        return 0
    asyncio.run(run_simulation(cancel=args.cancel))
    return 0


if __name__ == "__main__":
    os.environ["UNIVAI_MODE"] = "standalone"
    raise SystemExit(main())
