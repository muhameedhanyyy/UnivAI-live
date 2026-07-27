from __future__ import annotations

import asyncio
import json
import os
import unittest

os.environ["UNIVAI_MODE"] = "standalone"

from protocol import parse_room_name, validate_inbound, validate_script
from runtime import REPOSITORY_ROOT
from simulate import fixture_answer, run_simulation


class LiveSimulatorTests(unittest.TestCase):
    def test_room_name(self) -> None:
        self.assertEqual(("S-2026-000042", 3), parse_room_name("lecture-S-2026-000042-week-3"))
        with self.assertRaises(ValueError):
            parse_room_name("wrong-room")

    def test_fixture_and_protocol(self) -> None:
        script = json.loads(
            (REPOSITORY_ROOT / "fixtures" / "lecture.json").read_text(encoding="utf-8")
        )
        validate_script(script)
        with self.assertRaises(ValueError):
            validate_inbound({"type": "question", "text": ""})

    def test_complete_interaction_trace(self) -> None:
        messages = asyncio.run(run_simulation(trace=False))
        self.assertEqual("ended", messages[-1]["state"])
        self.assertTrue(any(item["type"] == "answer" for item in messages))

    def test_known_and_out_of_scope_qa(self) -> None:
        known = asyncio.run(fixture_answer("How is tenant material protected?"))
        unknown = asyncio.run(fixture_answer("What is the weather?"))
        self.assertEqual([2], known["pages"])
        self.assertEqual([], unknown["pages"])


if __name__ == "__main__":
    unittest.main()
