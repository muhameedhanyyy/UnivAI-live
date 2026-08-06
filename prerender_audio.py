"""Pre-render the lecturer's voice to disk, so lectures never wait on TTS.

    python UnivAI-live/prerender_audio.py   (from the UnivAI campus root)

For every week this renders each sentence of script.json to
lectures/week-N/audio/s{segment}-t{sentence}.npy (float32 PCM) plus a
meta.json with the sample rate. Personalized raise-hand prompts are produced
through the authenticated prompt-cache prewarm flow. Joining a lecture
costs a disk read, not a model load — and a machine too starved to load
Kokoro can still hold a full lecture.

Live TTS remains only for what cannot be known in advance: spoken answers.
"""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np

from campus_imports import configure_campus_imports

# Lecture titles land in log prints; a redirected Windows stdout is cp1252.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

configure_campus_imports()

from common.sentences import split_sentences  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = ROOT / "lectures"


def _valid_clip(path: Path) -> bool:
    try:
        clip = np.load(path, allow_pickle=False)
        return clip.dtype == np.float32 and clip.ndim == 1 and clip.size > 0
    except Exception:
        return False


def _remove_stale_clips(audio_dir: Path, expected: list[Path]) -> None:
    expected_paths = {path.resolve() for path in expected}
    for stale in audio_dir.glob("*.npy"):
        if stale.resolve() not in expected_paths:
            stale.unlink(missing_ok=True)


def _report_progress(
    book_id: int | None,
    week: int,
    total_weeks: int | None,
    ready_clips: int,
    total_clips: int,
) -> None:
    if book_id is None:
        return
    from common.db import execute

    message = (
        f"Recording lecture audio {week} of {total_weeks or '?'} — "
        f"{ready_clips}/{total_clips} clips ready…"
    )
    execute(
        """WITH touched_book AS (
             UPDATE books SET progress = %s, heartbeat_at = CURRENT_TIMESTAMP
             WHERE id = %s RETURNING id
           )
           UPDATE course_generation_milestones
           SET progress = %s, updated_at = CURRENT_TIMESTAMP
           WHERE book_id = %s AND week = %s AND stage = 'audio'""",
        (message, book_id, message, book_id, week),
    )


def prerender_all(
    sid: str | None = None,
    week: int | None = None,
    log=print,
    *,
    book_id: int | None = None,
    total_weeks: int | None = None,
) -> dict:
    from tts import load_engine  # imports onnx models — keep at call time

    engine = None
    sample_rate = None
    rendered = 0
    reused = 0

    # Multi-tenant: render this student's course under lectures/<sid>/week-N/.
    # Without a sid, fall back to the legacy global lectures/week-N/.
    base = LECTURES_DIR / sid if sid else LECTURES_DIR
    for folder in sorted(base.glob("week-*")):
        if week is not None and folder.name != f"week-{week}":
            continue
        script_path = folder / "script.json"
        if not script_path.exists():
            continue
        script = json.loads(script_path.read_text("utf-8"))
        script_sha256 = hashlib.sha256(script_path.read_bytes()).hexdigest()
        audio_dir = folder / "audio"
        audio_dir.mkdir(exist_ok=True)
        expected = [
            audio_dir / f"s{s_index}-t{t_index}.npy"
            for s_index, segment in enumerate(script["segments"])
            for t_index, _sentence in enumerate(split_sentences(segment["text"]))
        ]
        meta_path = audio_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text("utf-8"))
            except Exception:
                meta = {}
            clips_complete = bool(expected) and all(_valid_clip(path) for path in expected)
            if meta.get("script_sha256") == script_sha256 and clips_complete:
                _remove_stale_clips(audio_dir, expected)
                sample_rate = meta.get("sample_rate", sample_rate)
                _report_progress(book_id, week or 0, total_weeks, len(expected), len(expected))
                log(f"[prerender] {folder.name}: already complete")
                continue
            if not meta.get("script_sha256") and clips_complete:
                _remove_stale_clips(audio_dir, expected)
                meta["script_sha256"] = script_sha256
                temporary_meta = audio_dir / ".meta.json.tmp"
                temporary_meta.write_text(json.dumps(meta), encoding="utf-8")
                temporary_meta.replace(meta_path)
                sample_rate = meta.get("sample_rate", sample_rate)
                _report_progress(book_id, week or 0, total_weeks, len(expected), len(expected))
                log(f"[prerender] {folder.name}: adopted complete legacy audio")
                continue
            if meta.get("script_sha256") != script_sha256:
                for stale in audio_dir.glob("*.npy"):
                    stale.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)

        ready_count = sum(1 for path in expected if _valid_clip(path))
        _report_progress(book_id, week or 0, total_weeks, ready_count, len(expected))
        log(f"[prerender] {folder.name}: {script['title']}")
        if engine is None:
            engine = load_engine()
            sample_rate = engine.sample_rate
        for s_index, segment in enumerate(script["segments"]):
            for t_index, sentence in enumerate(split_sentences(segment["text"])):
                target = audio_dir / f"s{s_index}-t{t_index}.npy"
                if target.exists() and _valid_clip(target):
                    reused += 1
                    continue
                temporary = target.with_name(f".{target.name}.tmp.npy")
                np.save(temporary, engine.render(sentence).astype(np.float32))
                temporary.replace(target)
                rendered += 1
                ready_count += 1
                if ready_count % 10 == 0 or ready_count == len(expected):
                    _report_progress(book_id, week or 0, total_weeks, ready_count, len(expected))

        _remove_stale_clips(audio_dir, expected)

        temporary_meta = audio_dir / ".meta.json.tmp"
        temporary_meta.write_text(
            json.dumps({"sample_rate": engine.sample_rate, "script_sha256": script_sha256}),
            encoding="utf-8",
        )
        temporary_meta.replace(meta_path)

    log(f"[prerender] done: {rendered} new, {reused} reused at {sample_rate or 0} Hz")
    return {"ok": True, "clips": rendered, "reused": reused, "sample_rate": sample_rate or 0}


if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else None
    week = int(sys.argv[2]) if len(sys.argv) > 2 else None
    book_id = int(sys.argv[3]) if len(sys.argv) > 3 else None
    total_weeks = int(sys.argv[4]) if len(sys.argv) > 4 else None
    try:
        print(
            json.dumps(
                prerender_all(
                    sid,
                    week,
                    book_id=book_id,
                    total_weeks=total_weeks,
                )
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        raise SystemExit(1)
