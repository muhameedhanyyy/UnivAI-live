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
    # Every filter is optional, so each placeholder needs its type spelled out:
    # a bare NULL leaves Postgres unable to infer one and the whole query fails
    # with IndeterminateDatatype before a single sentence is rendered.
    rows = fetch_all(
        """SELECT artifact_id::text AS artifact_id, student_id, week,
                  script_payload
             FROM lecture_artifacts
            WHERE (%s::text IS NULL OR student_id = %s::text)
              AND (%s::int IS NULL OR week = %s::int)
              AND (%s::int IS NULL OR book_id = %s::int)
            ORDER BY student_id, week""",
        (sid, sid, week, week, book_id, book_id),
    )
    cache = AudioCache()
    rendered = reused = 0
    # Loading the TTS model costs seconds and a lot of memory. A course adopted
    # from another learner is already entirely cached, so the engine is loaded
    # on the first genuine miss and never at all for a pure warm-up.
    engine = None
    for row in rows:
        script = row["script_payload"]
        digest = script_digest(script)
        for segment_index, segment in enumerate(script["segments"]):
            for sentence_index, sentence in enumerate(split_sentences(segment["text"])):
                if cache.load(digest, segment_index, sentence_index) is not None:
                    reused += 1
                    continue
                if engine is None:
                    engine = load_live_engine()
                audio = engine.render(sentence)
                cache.store(
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
        "sample_rate": engine.sample_rate if engine is not None else None,
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
