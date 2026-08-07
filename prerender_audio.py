"""Pre-fill the disposable narration cache from database lecture artifacts."""

from __future__ import annotations

import argparse
import json

from audio_cache import AudioCache, script_digest
from campus_imports import configure_campus_imports

configure_campus_imports()

from common.db import fetch_all  # noqa: E402
from common.sentences import split_sentences  # noqa: E402
from tts import load_live_engine  # noqa: E402


def prerender_all(
    sid: str | None = None,
    week: int | None = None,
    *,
    book_id: int | None = None,
    log=print,
) -> dict:
    rows = fetch_all(
        """SELECT artifact_id::text AS artifact_id, student_id, week,
                  script_payload
             FROM lecture_artifacts
            WHERE (%s IS NULL OR student_id = %s)
              AND (%s IS NULL OR week = %s)
              AND (%s IS NULL OR book_id = %s)
            ORDER BY student_id, week""",
        (sid, sid, week, week, book_id, book_id),
    )
    engine = load_live_engine()
    cache = AudioCache()
    rendered = reused = 0
    for row in rows:
        script = row["script_payload"]
        digest = script_digest(script)
        artifact_id = str(row["artifact_id"])
        for segment_index, segment in enumerate(script["segments"]):
            for sentence_index, sentence in enumerate(split_sentences(segment["text"])):
                if cache.load(artifact_id, digest, segment_index, sentence_index) is not None:
                    reused += 1
                    continue
                audio = engine.render(sentence)
                cache.store(
                    artifact_id,
                    digest,
                    segment_index,
                    sentence_index,
                    audio,
                    engine.sample_rate,
                )
                rendered += 1
        log(f"[prerender] {row['student_id']} week {row['week']} ready")
    result = {
        "ok": True,
        "lectures": len(rows),
        "rendered": rendered,
        "reused": reused,
        "sample_rate": engine.sample_rate,
    }
    log(json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student")
    parser.add_argument("--week", type=int)
    parser.add_argument("--book", type=int)
    args = parser.parse_args()
    prerender_all(args.student, args.week, book_id=args.book)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
