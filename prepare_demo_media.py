"""Build and preflight deterministic browser media for the no-LiveKit demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import uuid
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from dotenv import load_dotenv

from audio_cache import AudioCache, script_digest
from campus_imports import configure_campus_imports

configure_campus_imports()

from common.db import fetch_all  # noqa: E402
from common.sentences import split_sentences  # noqa: E402
from personalization import render_templates  # noqa: E402
from tts import load_live_engine  # noqa: E402

CAMPUS_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(CAMPUS_ROOT / ".env")
_media_root = Path(os.getenv("DEMO_MEDIA_ROOT", "demo-media")).expanduser()
MEDIA_ROOT = (_media_root if _media_root.is_absolute() else CAMPUS_ROOT / _media_root).resolve()
LECTURES_ROOT = (CAMPUS_ROOT / "lectures").resolve()
LECTURE_SCHEMA = "univai.demo-media.lecture"
SECTION_SCHEMA = "univai.demo-media.section"
VERSION = "1.0.0"
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.I)
SHA256 = re.compile(r"^[a-f0-9]{64}$")
SILENCE_MS = 120
WELCOME_BACK = "Welcome back. I will continue from three sentences before where we stopped."
FIRST_JOIN = "Welcome. Your lecture audio is ready."
SECTION_WELCOME = "Welcome back. We will continue from your saved section step."


class PreparationError(RuntimeError):
    pass


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def lecture_bundle_target(row: dict[str, Any], digest: str) -> Path:
    sid = str(row.get("student_id") or "")
    week = row.get("week")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", sid):
        raise PreparationError("lecture account folder is invalid")
    if isinstance(week, bool) or not isinstance(week, int) or week < 1:
        raise PreparationError("lecture week folder is invalid")
    return (
        LECTURES_ROOT
        / sid
        / f"week-{week}"
        / "demo-media"
        / row["artifact_id"]
        / f"plan-{row['plan_version']}"
        / digest
    )


def milliseconds(samples: int, sample_rate: int) -> int:
    return round(samples * 1000 / sample_rate)


def vtt_time(value_ms: int) -> str:
    hours, remainder = divmod(value_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def wav_info(path: Path) -> tuple[int, int, int]:
    try:
        with wave.open(str(path), "rb") as source:
            if source.getnchannels() != 1 or source.getsampwidth() != 2:
                raise PreparationError(f"{path.name} must be mono 16-bit PCM")
            frames, rate = source.getnframes(), source.getframerate()
    except (OSError, EOFError, wave.Error) as exc:
        raise PreparationError(f"{path.name} is not decodable WAV audio: {exc}") from exc
    if frames <= 0 or rate < 8_000:
        raise PreparationError(f"{path.name} is empty or has an invalid sample rate")
    return frames, rate, milliseconds(frames, rate)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not samples.size or sample_rate < 8_000 or not np.isfinite(samples).all():
        raise PreparationError("TTS returned empty or invalid audio")
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


@contextmanager
def publication_lock(target: Path, timeout_seconds: float = 30.0):
    lock = target.with_name(f".{target.name}.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"{os.getpid()}\n".encode())
        except FileExistsError:
            if target.exists():
                yield False
                return
            if time.monotonic() - started >= timeout_seconds:
                raise PreparationError(f"Timed out waiting for media lock {lock.name}")
            time.sleep(0.1)
    try:
        yield True
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock.unlink(missing_ok=True)


def atomic_directory(target: Path, builder) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        builder(temporary)
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


class Renderer:
    def __init__(self) -> None:
        self.cache = AudioCache()
        self.engine = None

    def _engine(self):
        if self.engine is None:
            self.engine = load_live_engine()
        return self.engine

    def sentence(self, digest: str, segment: int, sentence: int, text: str) -> tuple[np.ndarray, int]:
        cached = self.cache.load(digest, segment, sentence)
        if cached is not None:
            return cached
        engine = self._engine()
        audio = np.asarray(engine.render(text), dtype=np.float32)
        if not audio.size:
            raise PreparationError(f"TTS produced no audio for segment {segment}, sentence {sentence}")
        self.cache.store(digest, segment, sentence, audio, engine.sample_rate)
        return audio, int(engine.sample_rate)

    @staticmethod
    def clip_path(text: str) -> Path:
        normalized = " ".join(str(text).split())
        if not normalized:
            raise PreparationError("Static audio text is empty")
        digest = digest_bytes(normalized.encode("utf-8"))
        return MEDIA_ROOT / "content" / "clips" / digest[:2] / f"{digest}.wav"

    def clip(self, text: str) -> Path:
        normalized = " ".join(str(text).split())
        target = self.clip_path(normalized)
        if target.exists():
            wav_info(target)
            return target
        with publication_lock(target) as owner:
            if not owner or target.exists():
                wav_info(target)
                return target
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp.wav")
            try:
                engine = self._engine()
                write_wav(temporary, engine.render(normalized), int(engine.sample_rate))
                wav_info(temporary)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return target


def media_file(path: Path, relative: str, *, duration: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": relative.replace("\\", "/"),
        "mimeType": "audio/wav" if path.suffix == ".wav" else "text/vtt; charset=utf-8",
        "sha256": digest_file(path),
        "byteLength": path.stat().st_size,
    }
    if duration:
        result["durationMs"] = wav_info(path)[2]
    return result


def flatten_script(script: dict[str, Any]) -> list[dict[str, Any]]:
    segments = script.get("segments")
    if not isinstance(segments, list) or not segments:
        raise PreparationError("lecture script has no segments")
    result: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise PreparationError(f"script segment {segment_index} is invalid")
        slide = segment.get("slide")
        if isinstance(slide, bool) or not isinstance(slide, int) or slide < 1:
            raise PreparationError(f"script segment {segment_index} has an invalid slide")
        pages = sorted({
            citation.get("page")
            for citation in segment.get("citations", [])
            if isinstance(citation, dict) and isinstance(citation.get("page"), int) and citation["page"] > 0
        })
        for sentence_index, sentence in enumerate(split_sentences(str(segment.get("text") or ""))):
            result.append({
                "text": sentence,
                "slide": slide,
                "segmentIndex": segment_index,
                "sentenceIndex": sentence_index,
                "pages": pages,
            })
    if not result:
        raise PreparationError("lecture script has no speakable sentences")
    return result


def build_lecture_content(script: dict[str, Any], renderer: Renderer) -> tuple[Path, list[dict[str, Any]]]:
    digest = script_digest(script)
    target = MEDIA_ROOT / "content" / "lectures" / digest[:2] / digest
    audio_path = target / "lecture.wav"
    timing_path = target / "timings.json"
    expected = flatten_script(script)
    if target.exists():
        timings = json.loads(timing_path.read_text("utf-8"))
        wav_info(audio_path)
        if [item["text"] for item in timings] != [item["text"] for item in expected]:
            raise PreparationError("content-addressed lecture timings do not match the script")
        return audio_path, timings
    with publication_lock(target) as owner:
        if not owner or target.exists():
            timings = json.loads(timing_path.read_text("utf-8"))
            return audio_path, timings

        def build(temporary: Path) -> None:
            sample_rate: int | None = None
            cursor = 0
            chunks: list[np.ndarray] = []
            timings: list[dict[str, Any]] = []
            for source in expected:
                audio, rate = renderer.sentence(
                    digest,
                    source["segmentIndex"],
                    source["sentenceIndex"],
                    source["text"],
                )
                if sample_rate is None:
                    sample_rate = rate
                if rate != sample_rate:
                    raise PreparationError("TTS sample rate changed within one lecture")
                silence = np.zeros(round(rate * SILENCE_MS / 1000), dtype=np.float32)
                start = cursor
                chunks.extend((audio, silence))
                cursor += len(audio) + len(silence)
                timings.append({**source, "startSample": start, "endSample": cursor, "text": source["text"]})
            if not sample_rate:
                raise PreparationError("lecture produced no audio")
            write_wav(temporary / "lecture.wav", np.concatenate(chunks), sample_rate)
            (temporary / "timings.json").write_bytes(json_bytes(timings))
            frames, actual_rate, _ = wav_info(temporary / "lecture.wav")
            if actual_rate != sample_rate or frames != cursor:
                raise PreparationError("encoded lecture boundaries do not match the final WAV")

        atomic_directory(target, build)
    return audio_path, json.loads(timing_path.read_text("utf-8"))


def validate_lecture_bundle(target: Path, row: dict[str, Any]) -> None:
    manifest = json.loads((target / "manifest.json").read_text("utf-8"))
    if manifest.get("schema") != LECTURE_SCHEMA or manifest.get("version") != VERSION:
        raise PreparationError("lecture manifest schema is invalid")
    if manifest.get("studentId") != row["student_id"] or manifest.get("lecturePublicId") != row["public_id"] or manifest.get("artifactId") != row["artifact_id"] or manifest.get("planVersion") != row["plan_version"]:
        raise PreparationError("lecture manifest identity is stale")
    if manifest.get("scriptDigest") != script_digest(row["script_payload"]):
        raise PreparationError("lecture manifest script digest is stale")
    cues = manifest.get("cues")
    expected = flatten_script(row["script_payload"])
    if not isinstance(cues, list) or len(cues) != len(expected):
        raise PreparationError("lecture cue count does not match the script")
    frames, rate, duration = wav_info(target / manifest["audio"]["path"])
    if duration != manifest["audio"]["durationMs"] or frames <= 0:
        raise PreparationError("lecture audio duration is invalid")
    previous_end = -1
    vtt_blocks: list[tuple[str, str, str]] = []
    lines = (target / manifest["captions"]["path"]).read_text("utf-8").replace("\r\n", "\n").strip().split("\n\n")
    if not lines or lines[0] != "WEBVTT":
        raise PreparationError("lecture VTT header is invalid")
    for block in lines[1:]:
        parts = block.splitlines()
        if len(parts) < 3:
            raise PreparationError("lecture VTT cue is malformed")
        vtt_blocks.append((parts[0], parts[1], "\n".join(parts[2:])))
    if len(vtt_blocks) != len(cues):
        raise PreparationError("lecture VTT cue count is invalid")
    # Slide 1 is the deterministic cover created by render_slidev_markdown;
    # slides_payload contains the generated teaching slides starting at 2.
    slides = {1, *(
        slide.get("slide")
        for slide in row["slides_payload"].get("slides", [])
        if isinstance(slide, dict)
    )}
    for index, (cue, source, block) in enumerate(zip(cues, expected, vtt_blocks, strict=True)):
        expected_id = f"slide-{source['slide']}.segment-{source['segmentIndex']}.sentence-{source['sentenceIndex']}"
        if cue.get("id") != expected_id or cue.get("flatCueIndex") != index or cue.get("text") != source["text"]:
            raise PreparationError(f"lecture cue {index} is inconsistent")
        start, end = cue.get("startMs"), cue.get("endMs")
        if not isinstance(start, int) or not isinstance(end, int) or start < previous_end or end <= start or end > duration + 2:
            raise PreparationError(f"lecture cue {index} timing is invalid")
        if source["slide"] not in slides:
            raise PreparationError(f"lecture cue {index} references a missing slide")
        if block != (expected_id, f"{vtt_time(start)} --> {vtt_time(end)}", source["text"]):
            raise PreparationError(f"lecture VTT cue {index} disagrees with the manifest")
        previous_end = end
    for key in ("audio", "captions", "welcomeBack", "firstJoin"):
        media = manifest[key]
        file = target / media["path"]
        if not file.is_file() or file.stat().st_size != media["byteLength"] or digest_file(file) != media["sha256"]:
            raise PreparationError(f"lecture {key} integrity check failed")
        if key != "captions" and wav_info(file)[2] != media.get("durationMs"):
            raise PreparationError(f"lecture {key} duration or decoding check failed")


def prepare_lecture(row: dict[str, Any], renderer: Renderer, preflight: bool) -> str:
    digest = script_digest(row["script_payload"])
    target = lecture_bundle_target(row, digest)
    if target.exists():
        validate_lecture_bundle(target, row)
        return "READY" if preflight else "REUSED"
    if preflight:
        raise PreparationError("demo audio is missing")
    audio, timings = build_lecture_content(row["script_payload"], renderer)
    welcome = renderer.clip(WELCOME_BACK)
    first_join = renderer.clip(FIRST_JOIN)

    def build(temporary: Path) -> None:
        link_or_copy(audio, temporary / "lecture.wav")
        link_or_copy(welcome, temporary / "welcome-back.wav")
        link_or_copy(first_join, temporary / "first-join.wav")
        _, sample_rate, duration_ms = wav_info(temporary / "lecture.wav")
        cues: list[dict[str, Any]] = []
        vtt = ["WEBVTT", ""]
        for flat_index, timing in enumerate(timings):
            start_ms = milliseconds(timing["startSample"], sample_rate)
            end_ms = milliseconds(timing["endSample"], sample_rate)
            cue_id = f"slide-{timing['slide']}.segment-{timing['segmentIndex']}.sentence-{timing['sentenceIndex']}"
            cue = {
                "id": cue_id,
                "startMs": start_ms,
                "endMs": end_ms,
                "slide": timing["slide"],
                "segmentIndex": timing["segmentIndex"],
                "sentenceIndex": timing["sentenceIndex"],
                "flatCueIndex": flat_index,
                "text": timing["text"],
                "pages": timing["pages"],
            }
            cues.append(cue)
            vtt.extend((cue_id, f"{vtt_time(start_ms)} --> {vtt_time(end_ms)}", timing["text"], ""))
        (temporary / "lecture.vtt").write_text("\n".join(vtt), encoding="utf-8", newline="\n")
        manifest = {
            "schema": LECTURE_SCHEMA,
            "version": VERSION,
            "studentId": row["student_id"],
            "lecturePublicId": row["public_id"],
            "artifactId": row["artifact_id"],
            "planVersion": row["plan_version"],
            "scriptDigest": digest,
            "audio": media_file(temporary / "lecture.wav", "lecture.wav", duration=True),
            "captions": media_file(temporary / "lecture.vtt", "lecture.vtt", duration=False),
            "welcomeBack": media_file(temporary / "welcome-back.wav", "welcome-back.wav", duration=True),
            "firstJoin": media_file(temporary / "first-join.wav", "first-join.wav", duration=True),
            "sourcePages": sorted({page for cue in cues for page in cue["pages"]}),
            "slideMapping": list(dict.fromkeys(cue["slide"] for cue in cues)),
            "cues": cues,
        }
        (temporary / "manifest.json").write_bytes(json_bytes(manifest))
        validate_lecture_bundle(temporary, row)

    with publication_lock(target) as owner:
        if not owner or target.exists():
            validate_lecture_bundle(target, row)
            return "REUSED"
        atomic_directory(target, build)
    return "REGENERATED"


def item_text(item: dict[str, Any], *keys: str) -> str:
    return " ".join(str(item.get(key) or "").strip() for key in keys if str(item.get(key) or "").strip()).strip()


def section_nodes(pack: dict[str, Any]) -> list[dict[str, Any]]:
    if pack.get("schema_name") != "univai.section.pack" or pack.get("schema_version") != "1.0.0" or pack.get("session_type") != "section":
        raise PreparationError("section pack schema is invalid")
    activities = pack.get("activities")
    if not isinstance(activities, list) or not activities:
        raise PreparationError("section pack has no activities")
    nodes: list[dict[str, Any]] = []
    objectives = pack.get("objectives") if isinstance(pack.get("objectives"), list) else []
    intro = ". ".join([str(pack.get("title") or "Section practice"), *[str(value) for value in objectives if str(value).strip()]])
    nodes.append({"state": "intro", "activityIndex": None, "stepIndex": None, "title": str(pack.get("title") or "Section"), "text": intro, "citations": []})
    examples = pack.get("examples") if isinstance(pack.get("examples"), list) else []
    for example_index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise PreparationError(f"section example {example_index} is invalid")
        prompt = item_text(example, "prompt", "description")
        if prompt:
            nodes.append({"state": "example", "activityIndex": example_index, "stepIndex": None, "title": str(example.get("title") or f"Example {example_index + 1}"), "text": prompt, "citations": example.get("citations", [])})
        steps = example.get("steps") if isinstance(example.get("steps"), list) else []
        for step_index, step in enumerate(steps):
            if not isinstance(step, dict):
                raise PreparationError(f"section example step {step_index} is invalid")
            spoken = item_text(step, "step", "explanation", "conclusion")
            if not spoken:
                raise PreparationError(f"section example step {step_index} has no audible text")
            nodes.append({"state": "example", "activityIndex": example_index, "stepIndex": step_index, "title": str(step.get("step") or f"Step {step_index + 1}"), "text": spoken, "citations": step.get("citations", [])})
    for activity_index, activity in enumerate(activities):
        if not isinstance(activity, dict):
            raise PreparationError(f"section activity {activity_index} is invalid")
        spoken = item_text(activity, "title", "description", "prompt")
        if not spoken:
            raise PreparationError(f"section activity {activity_index} has no audible text")
        nodes.append({"state": "guided_task", "activityIndex": activity_index, "stepIndex": None, "title": str(activity.get("title") or f"Activity {activity_index + 1}"), "text": spoken, "citations": activity.get("citations", [])})
    todos = pack.get("todos") if isinstance(pack.get("todos"), list) else []
    if todos:
        recap = ". ".join(item_text(todo, "text", "title", "description") for todo in todos if isinstance(todo, dict))
        if recap:
            nodes.append({"state": "todo_recap", "activityIndex": None, "stepIndex": None, "title": "Next actions", "text": recap, "citations": [citation for todo in todos if isinstance(todo, dict) for citation in todo.get("citations", [])]})
    return nodes


def validate_section_bundle(target: Path, row: dict[str, Any]) -> None:
    manifest = json.loads((target / "manifest.json").read_text("utf-8"))
    if manifest.get("schema") != SECTION_SCHEMA or manifest.get("version") != VERSION:
        raise PreparationError("section manifest schema is invalid")
    if manifest.get("sectionPackId") != row["section_pack_id"] or manifest.get("payloadHash") != row["payload_hash"] or manifest.get("planVersion") != row["plan_version"]:
        raise PreparationError("section manifest identity is stale")
    expected = section_nodes(row["pack_payload"])
    nodes = manifest.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != len(expected):
        raise PreparationError("section manifest does not cover every static node")
    for index, (node, source) in enumerate(zip(nodes, expected, strict=True)):
        if node.get("id") != f"node-{index:03d}" or node.get("text") != source["text"] or node.get("state") != source["state"]:
            raise PreparationError(f"section node {index} is inconsistent")
        media = node.get("audio", {})
        file = target / str(media.get("path") or "")
        if not file.is_file() or digest_file(file) != media.get("sha256") or wav_info(file)[2] != media.get("durationMs"):
            raise PreparationError(f"section node {index} audio is invalid")
    welcome = manifest.get("welcomeBack", {})
    welcome_file = target / str(welcome.get("path") or "")
    if not welcome_file.is_file() or digest_file(welcome_file) != welcome.get("sha256") or wav_info(welcome_file)[2] != welcome.get("durationMs"):
        raise PreparationError("section welcome audio is invalid")


def prepare_section(row: dict[str, Any], renderer: Renderer, preflight: bool) -> str:
    target = MEDIA_ROOT / "sections" / row["section_pack_id"] / f"plan-{row['plan_version']}" / row["payload_hash"]
    if target.exists():
        validate_section_bundle(target, row)
        return "READY" if preflight else "REUSED"
    if preflight:
        raise PreparationError("demo audio is missing")
    nodes = section_nodes(row["pack_payload"])
    welcome = renderer.clip(SECTION_WELCOME)
    clips = [renderer.clip(node["text"]) for node in nodes]

    def build(temporary: Path) -> None:
        link_or_copy(welcome, temporary / "welcome-back.wav")
        manifest_nodes = []
        for index, (node, source) in enumerate(zip(nodes, clips, strict=True)):
            relative = f"clips/node-{index:03d}.wav"
            link_or_copy(source, temporary / relative)
            manifest_nodes.append({"id": f"node-{index:03d}", **node, "audio": media_file(temporary / relative, relative, duration=True)})
        manifest = {
            "schema": SECTION_SCHEMA,
            "version": VERSION,
            "sectionPackId": row["section_pack_id"],
            "planVersion": row["plan_version"],
            "payloadHash": row["payload_hash"],
            "welcomeBack": media_file(temporary / "welcome-back.wav", "welcome-back.wav", duration=True),
            "nodes": manifest_nodes,
        }
        (temporary / "manifest.json").write_bytes(json_bytes(manifest))
        validate_section_bundle(temporary, row)

    with publication_lock(target) as owner:
        if not owner or target.exists():
            validate_section_bundle(target, row)
            return "REUSED"
        atomic_directory(target, build)
    return "REGENERATED"


def resolve_students(email: str | None, student: str | None, all_current: bool) -> list[dict[str, Any]]:
    if email:
        rows = fetch_all(
            'SELECT "email", "name", "registrationNumber" AS sid FROM "user" WHERE lower("email") = lower(%s)',
            (email.strip(),),
        )
    elif student:
        rows = fetch_all(
            'SELECT "email", "name", "registrationNumber" AS sid FROM "user" WHERE "registrationNumber" = %s',
            (student.strip(),),
        )
    elif all_current:
        rows = fetch_all(
            '''SELECT DISTINCT u."email", u."name", u."registrationNumber" AS sid
                 FROM "user" AS u
                 JOIN programmes AS p ON p.student_id = u."registrationNumber" AND p.status = 'approved'
                WHERE u."registrationNumber" IS NOT NULL
                ORDER BY u."registrationNumber"'''
        )
    else:
        raise PreparationError("choose --email, --student, or --all-current")
    rows = [row for row in rows if isinstance(row.get("sid"), str) and row["sid"].strip()]
    if not rows and all_current:
        return []
    if not rows:
        raise PreparationError("No matching existing account with a registration number was found")
    return rows


def approved_programme(sid: str) -> dict[str, Any]:
    programmes = fetch_all(
        "SELECT id::text AS id, status, plan_version FROM programmes WHERE student_id = %s ORDER BY id DESC",
        (sid,),
    )
    approved = next((row for row in programmes if row.get("status") == "approved"), None)
    if approved is not None:
        return approved
    if not programmes:
        raise PreparationError(f"{sid}: no programme record exists")
    statuses = ", ".join(str(row.get("status") or "unknown") for row in programmes)
    raise PreparationError(f"{sid}: no approved programme exists (found: {statuses})")


def lecture_rows(sid: str, week: int | None) -> list[dict[str, Any]]:
    programme = approved_programme(sid)
    rows = fetch_all(
        '''SELECT l.public_id::text AS public_id, l.id AS internal_id, l.week, l.title,
                  l.lecture_artifact_id::text AS linked_artifact_id,
                  la.artifact_id::text AS artifact_id, la.script_payload, la.slides_payload
             FROM lectures AS l
             LEFT JOIN lecture_artifacts AS la ON la.artifact_id = l.lecture_artifact_id
            WHERE l.student_id = %s
              AND (%s::integer IS NULL OR l.week = %s::integer)
            ORDER BY l.week''',
        (sid, week, week),
    )
    if not rows:
        selector = f"week {week}" if week is not None else "the current account"
        raise PreparationError(f"{sid}: no lecture record exists for {selector}")
    for row in rows:
        label = f"{sid} week {row.get('week')}"
        if not UUID.fullmatch(str(row.get("public_id") or "")):
            raise PreparationError(f"{label}: lecture public ID is missing or invalid")
        if not row.get("linked_artifact_id"):
            raise PreparationError(f"{label}: lecture_artifact_id is missing")
        if not UUID.fullmatch(str(row.get("artifact_id") or "")):
            raise PreparationError(f"{label}: referenced lecture artifact does not exist")
        if not isinstance(row.get("script_payload"), dict):
            raise PreparationError(f"{label}: narration script payload is missing")
        if not isinstance(row.get("slides_payload"), dict):
            raise PreparationError(f"{label}: slides payload is missing")
        row["programme_id"] = programme["id"]
        row["plan_version"] = programme["plan_version"]
        row["student_id"] = sid
    return rows


def section_rows(sid: str, week: int | None) -> list[dict[str, Any]]:
    programme = approved_programme(sid)
    rows = fetch_all(
        '''SELECT sp.section_pack_id, sp.week, sp.approved_plan_version AS plan_version,
                  sp.payload_hash, sp.pack_payload
             FROM section_packs AS sp
            WHERE sp.tenant_id = %s
              AND sp.programme_id = %s
              AND sp.approved_plan_version = %s
              AND (%s::integer IS NULL OR sp.week = %s::integer)
            ORDER BY sp.week''',
        (sid, programme["id"], programme["plan_version"], week, week),
    )
    if not rows:
        selector = f"week {week}" if week is not None else "the current approved plan"
        raise PreparationError(f"{sid}: no section pack exists for {selector}")
    for row in rows:
        if not UUID.fullmatch(str(row.get("section_pack_id") or "")) or not SHA256.fullmatch(str(row.get("payload_hash") or "")):
            raise PreparationError(f"{sid} week {row.get('week')}: section identity is invalid")
        if not isinstance(row.get("pack_payload"), dict):
            raise PreparationError(f"{sid} week {row.get('week')}: section pack is missing")
    return rows


def run(args: argparse.Namespace) -> int:
    renderer = Renderer()
    failures = 0
    try:
        students = resolve_students(args.email, args.student, args.all_current)
    except Exception as exc:
        print(f"FAILED account: {exc}")
        return 1
    for account in students:
        sid = account["sid"]
        lectures: list[dict[str, Any]] = []
        sections: list[dict[str, Any]] = []
        try:
            prompts = render_templates(str(account.get("name") or ""))
            for key in ("ask", "answer_resume"):
                target = renderer.clip_path(prompts[key])
                if args.preflight:
                    wav_info(target)
                else:
                    renderer.clip(prompts[key])
            print(f"{'READY' if args.preflight else 'PREPARED'} {sid} Piper interaction voice")
        except Exception as exc:
            failures += 1
            print(f"FAILED {sid} Piper interaction voice: {exc}")
        try:
            lectures = lecture_rows(sid, args.week)
        except Exception as exc:
            failures += 1
            print(f"FAILED {sid} lectures: {exc}")
        try:
            sections = section_rows(sid, args.week)
        except Exception as exc:
            failures += 1
            print(f"FAILED {sid} sections: {exc}")
        for row in lectures:
            label = f"{sid} lecture week {row['week']}"
            try:
                print(f"{prepare_lecture(row, renderer, args.preflight)} {label}")
            except Exception as exc:
                failures += 1
                print(f"FAILED {label}: {exc}")
        for row in sections:
            label = f"{sid} section week {row['week']}"
            try:
                print(f"{prepare_section(row, renderer, args.preflight)} {label}")
            except Exception as exc:
                failures += 1
                print(f"FAILED {label}: {exc}")
    print(json.dumps({"ok": failures == 0, "accounts": len(students), "failures": failures, "mediaRoot": str(MEDIA_ROOT)}, separators=(",", ":")))
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--email")
    selector.add_argument("--student")
    selector.add_argument("--all-current", action="store_true")
    parser.add_argument("--week", type=int)
    parser.add_argument("--preflight", action="store_true", help="validate only; never load TTS")
    args = parser.parse_args()
    if args.week is not None and args.week < 1:
        parser.error("--week must be positive")
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
