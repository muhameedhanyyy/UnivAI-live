"""Fail-closed demo preflight for credentials, models, ports and smoke Q&A."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from health import dependency_health  # noqa: E402


def run(*, smoke: bool) -> tuple[bool, list[dict]]:
    checks = [item.__dict__ for item in dependency_health()]
    if smoke:
        checks.append({"name": "smoke_qa", "ready": False, "reason": "run through configured integration harness"})
    return all(item["ready"] for item in checks), checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="require the externally configured smoke Q&A gate")
    args = parser.parse_args()
    ready, checks = run(smoke=args.smoke)
    print(json.dumps({"ready": ready, "checks": checks}, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
