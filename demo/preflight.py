"""Fail-closed demo preflight for credentials, models, ports and smoke Q&A."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from health import dependency_health  # noqa: E402


def run(*, smoke: bool) -> tuple[bool, list[dict]]:
    checks = [item.__dict__ for item in dependency_health()]
    checks.extend(_configuration_checks())
    if smoke:
        checks.append(_smoke_qa() if all(item["ready"] for item in checks) else {"name": "smoke_qa", "ready": False, "reason": "dependencies_not_ready"})
    return all(item["ready"] for item in checks), checks


def _configuration_checks() -> list[dict]:
    checks = []
    for name in ("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        checks.append({"name": name.lower(), "ready": bool(os.getenv(name)), "reason": "configured" if os.getenv(name) else "not_configured"})
    for name, value in (("livekit_port", os.getenv("LIVEKIT_URL")), ("agent_port", os.getenv("RAG_MCP_URL") or os.getenv("RAG_URL") or os.getenv("AGENT_URL"))):
        checks.append(_port_check(name, value))
    return checks


def _port_check(name: str, value: str | None) -> dict:
    from urllib.parse import urlparse
    if not value:
        return {"name": name, "ready": False, "reason": "not_configured"}
    parsed = urlparse(value if "://" in value else f"tcp://{value}")
    if not parsed.hostname:
        return {"name": name, "ready": False, "reason": "invalid_endpoint"}
    port = parsed.port or (443 if parsed.scheme in {"https", "wss"} else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=2):
            return {"name": name, "ready": True, "reason": "reachable"}
    except (OSError, TypeError):
        return {"name": name, "ready": False, "reason": "unreachable"}


def _smoke_qa() -> dict:
    question, learner_id = os.getenv("PREFLIGHT_SMOKE_QUESTION", "").strip(), os.getenv("PREFLIGHT_SMOKE_USER_ID", "").strip()
    if not question or not learner_id:
        return {"name": "smoke_qa", "ready": False, "reason": "question_or_user_not_configured"}
    try:
        from qa import TROUBLE, answer_question
        result = asyncio.run(answer_question(question, lecture_id=None, sid=learner_id))
        grounded = result.get("answer") != TROUBLE and bool(result.get("citations"))
        return {"name": "smoke_qa", "ready": grounded, "reason": "grounded_answer" if grounded else "no_grounded_answer"}
    except Exception as exc:
        return {"name": "smoke_qa", "ready": False, "reason": f"{type(exc).__name__}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-smoke", action="store_true", help="check dependencies only; never use this for the final demo gate")
    args = parser.parse_args()
    ready, checks = run(smoke=not args.skip_smoke)
    print(json.dumps({"ready": ready, "checks": checks}, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
