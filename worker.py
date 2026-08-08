"""The live lecture: one worker process, two agent identities in the LiveKit room.

    Lecturer  publishes the TTS audio track and drives the slides
    Listener   subscribes to the student's mic, runs VAD + STT, and interrupts

State machine (exactly as specified):

    LECTURING    stream the database script through TTS, sentence by sentence.
                 Each segment sends {type:"slide", n} so the Slidev iframe flips.
                 Student speaks (VAD >= 300 ms) -> INTERRUPTED

    INTERRUPTED  stop TTS immediately, remember the position (segment + sentence)
                 run STT until the student stops (silence ~800 ms) -> ANSWERING

    ANSWERING    question -> RAG (MCP) -> tiny LLM -> <=3 sentences, cited
                 speak the answer, then resume LECTURING from the remembered
                 sentence, restarting that sentence from its beginning.

    MUTED        the student's mic is muted client-side, so VAD never fires and
                 the lecture is never interrupted.

Run:  python UnivAI-live/worker.py dev   (from the UnivAI campus root)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from runtime import RuntimeMode, runtime_mode
from campus_imports import configure_campus_imports

if runtime_mode() is RuntimeMode.STANDALONE:
    raise RuntimeError(
        "worker.py is the integrated LiveKit entry point. "
        "Use 'python simulate.py run' for standalone mode."
    )

# Shared modules use both common.* and services.* package names internally.
configure_campus_imports()

import numpy as np
from dotenv import load_dotenv
from livekit import agents, rtc

from common.device import whisper_settings, describe  # noqa: E402
from common.sentences import split_sentences  # noqa: E402
from protocols.lecture_session import LectureSessionMeta, SessionMetadataError  # noqa: E402
from qa import TROUBLE, answer_question  # noqa: E402
from tts import load_live_engine  # noqa: E402
from prompt_cache import PromptCache  # noqa: E402
from startup import LazyDependencies, StartupStage, StartupTrace  # noqa: E402
from telemetry.startup_metrics import append_trace  # noqa: E402
from question_turn import QuestionTurnController, TurnState  # noqa: E402
from resilience.fallbacks import choose_fallback  # noqa: E402
from resilience.timeouts import Stage, StageTimeout, within_budget  # noqa: E402
from audio_cache import AudioCache, script_digest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# "localhost" resolves to ::1 first on Windows, and the Rust rtc client times
# out on a dropped ::1 before falling back to IPv4 — dial v4 loopback directly.
_url = os.getenv("LIVEKIT_URL", "")
if "://localhost" in _url:
    os.environ["LIVEKIT_URL"] = _url.replace("://localhost", "://127.0.0.1")

STT_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "base")

SPEECH_TRIGGER_MS = 300     # this much speech from the student = a barge-in
SILENCE_END_MS = 800        # this much silence = they have finished asking
REVIEW_TIMEOUT_S = 120      # how long we hold the lecture while they edit the transcript
PROMPT_TEXTS = {
    "ask": "Yes? Do you have a question? Unmute your microphone and go ahead.",
    "remind": "Your hand is still raised. Unmute whenever you are ready; I am listening.",
    "resume": "No question? No problem. Alright everyone, eyes back on the slides, and let us continue!",
}


def log(message: str) -> None:
    """print() with a wall-clock stamp: 'the speak took 4:35' is un-debuggable
    from prints that do not say WHEN. (Infra logging only — business logic
    still goes through the virtual clock.)"""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------- lecture script


@dataclass
class Position:
    """Where the Lecturer is in the script, so it can resume after a question."""

    segment: int = 0
    sentence: int = 0


@dataclass
class Lecture:
    week: int
    title: str
    segments: list[dict]
    # The owner (studentId). Scopes RAG retrieval for live Q&A and the qa_log.
    sid: str = ""
    artifact_id: str = ""
    script_digest: str = ""
    position: Position = field(default_factory=Position)

    @staticmethod
    def load(
        sid: str,
        week: int,
        artifact_id: str,
    ) -> "Lecture":
        from common.db import fetch_one
        row = fetch_one(
            """SELECT la.script_payload FROM lectures l
                 JOIN lecture_artifacts la ON la.artifact_id = l.lecture_artifact_id
                WHERE l.student_id = %s AND l.week = %s
                  AND la.artifact_id = %s::uuid""",
            (sid, week, artifact_id),
        )
        if not row or not isinstance(row.get("script_payload"), dict):
            raise FileNotFoundError("database lecture artifact is missing")
        script = row["script_payload"]
        segments = script.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError("database lecture has no narration segments")
        digest = script_digest(script)
        return Lecture(
            week=week,
            title=script["title"],
            segments=segments,
            sid=sid,
            artifact_id=artifact_id,
            script_digest=digest,
        )

# ---------------------------------------------------------------- the worker


class LectureSession:
    def __init__(self, room: rtc.Room, lecture: Lecture, tts, session_meta: LectureSessionMeta | None = None, *, startup_trace: StartupTrace | None = None, startup_mode: str = "warm", dependencies: LazyDependencies | None = None, audio_cache: AudioCache | None = None) -> None:
        self.room = room
        self.lecture = lecture
        self.session_meta = session_meta
        # Loaded in prewarm(). A missing engine leaves the data-channel text
        # usable, but cannot publish voice until the retry succeeds.
        self.tts = tts
        self._engine_retry = False
        self.startup_trace = startup_trace
        self.startup_mode = startup_mode
        self.dependencies = dependencies
        self.audio_cache = audio_cache
        self._first_frame_recorded = False

        # Engines differ: Piper is 22.05 kHz, XTTS is 24 kHz. Publishing at the
        # wrong rate does not fail — it just makes the lecturer sound wrong.
        self.sample_rate = tts.sample_rate if tts else 24000
        self.source = rtc.AudioSource(self.sample_rate, 1)
        self.track = rtc.LocalAudioTrack.create_audio_track("lecturer", self.source)

        self.interrupted = asyncio.Event()   # legacy stop signal (room closing)
        # The raise-hand protocol: the student asks permission BEFORE unmuting.
        self.hand_raised = asyncio.Event()
        self.mic_unmuted = asyncio.Event()
        self.hand_active = False             # capture window: only now does the Listener record
        self.prompts: dict[str, np.ndarray] = {}
        # What Whisper heard. It is shown in the browser for the student to correct.
        self.heard: asyncio.Queue[str] = asyncio.Queue()
        # What the student actually confirmed (possibly edited). "" means they cancelled.
        self.confirmed: asyncio.Queue[str] = asyncio.Queue()
        self.question_turn = QuestionTurnController()
        self.flush_capture = None
        self.speaking = False
        self.closed = False          # the student left; stop talking to an empty room

    # -- outbound messages to the browser --------------------------------------

    async def send(self, message: dict) -> None:
        if self.closed:
            return
        try:
            await self.room.local_participant.publish_data(
                json.dumps(message).encode("utf-8"), reliable=True
            )
        except Exception:
            self.closed = True

    # -- speaking ---------------------------------------------------------------

    def _fit(self, audio: np.ndarray, rate: int | None) -> np.ndarray:
        """Resample to the track's rate (linear — fine for speech)."""
        if not rate or rate == self.sample_rate or not len(audio):
            return audio
        length = int(len(audio) * self.sample_rate / rate)
        return np.interp(
            np.linspace(0, len(audio) - 1, length), np.arange(len(audio)), audio
        ).astype(np.float32)

    async def _engine(self):
        """The live TTS engine, or the best we can get. Kokoro can fail to load
        on a starved machine; Piper is 60 MB and almost always fits."""
        if self.tts is None and not self._engine_retry:
            self._engine_retry = True
            try:
                self.tts = await self.dependencies.tts() if self.dependencies else None
            except Exception as exc:
                print(f"[tts] no live engine available: {exc}")
        return self.tts

    async def render(self, text: str) -> np.ndarray:
        """Synthesis is CPU-bound, so keep it off the event loop."""
        engine = await self._engine()
        if engine is None:
            # Nothing can speak. The text still reaches the browser as data.
            return np.zeros(0, dtype=np.float32)
        try:
            audio = await within_budget(Stage.TTS, asyncio.to_thread(engine.render, text))
        except StageTimeout:
            fallback = choose_fallback("tts", "tts_timeout")
            await self.send(fallback.event())
            return np.zeros(0, dtype=np.float32)
        return self._fit(audio, engine.sample_rate)

    async def sentence_audio(self, segment: int, sentence: int, text: str) -> np.ndarray:
        """Use the shared render cache, synthesizing and populating it on miss."""
        if self.audio_cache is not None:
            cached = self.audio_cache.load(
                self.lecture.script_digest,
                segment,
                sentence,
            )
            if cached is not None:
                audio, rate = cached
                return self._fit(audio, rate)
        audio = await self.render(text)
        if self.audio_cache is not None and len(audio):
            await asyncio.to_thread(
                self.audio_cache.store,
                self.lecture.script_digest,
                segment,
                sentence,
                audio,
                self.sample_rate,
            )
        return audio

    async def play(self, audio: np.ndarray, interruptible: bool = True) -> bool:
        """Stream one rendered sentence. False = the student cut in, or the room died."""
        self.speaking = True
        frame_size = self.sample_rate // 10  # 100 ms, so a barge-in cuts in fast
        try:
            for start in range(0, len(audio), frame_size):
                if interruptible and self.interrupted.is_set():
                    return False
                if self.closed:
                    return False

                pcm = (np.clip(audio[start : start + frame_size], -1.0, 1.0) * 32767).astype(
                    np.int16
                )
                frame = rtc.AudioFrame(
                    data=pcm.tobytes(),
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    samples_per_channel=len(pcm),
                )
                try:
                    await self.source.capture_frame(frame)
                    if not self._first_frame_recorded and self.startup_trace:
                        self._first_frame_recorded = True
                        self.startup_trace.mark(StartupStage.FIRST_FRAME)
                        payload = self.startup_trace.payload(mode=self.startup_mode)
                        print(json.dumps(payload, sort_keys=True), flush=True)
                        trace_path = os.getenv("STARTUP_TRACE_PATH")
                        if trace_path:
                            append_trace(Path(trace_path), payload)
                        if self.dependencies:
                            asyncio.create_task(self._warm_after_first_frame())
                except Exception as exc:
                    # The student closed the tab mid-sentence. Stop talking to an
                    # empty room rather than crashing the worker.
                    print(f"[lecture] room is gone, stopping: {exc}")
                    self.closed = True
                    return False
            return True
        finally:
            self.speaking = False

    async def speak(self, text: str, interruptible: bool = True) -> bool:
        return await self.play(await self.render(text), interruptible=interruptible)

    async def _warm_after_first_frame(self) -> None:
        """Prepare interaction models and prompts after narration is audible."""
        results = await self.dependencies.warm()
        engine = results[0]
        if isinstance(engine, BaseException):
            print(f"[tts] deferred engine warmup failed: {engine}")
            return
        self.tts = engine
        rendered = await asyncio.gather(
            *(self.render(text) for text in PROMPT_TEXTS.values()),
            return_exceptions=True,
        )
        for key, audio in zip(PROMPT_TEXTS, rendered):
            if isinstance(audio, np.ndarray) and len(audio):
                self.prompts[key] = audio

    async def _play_prompt(self, key: str) -> None:
        audio = self.prompts.get(key)
        if audio is None:
            audio = await self.render(PROMPT_TEXTS[key])
            if len(audio):
                self.prompts[key] = audio
        if len(audio):
            await self.play(audio, interruptible=False)

    # -- the state machine ------------------------------------------------------

    async def run(self) -> None:
        timeout = self.startup_trace.remaining() if self.startup_trace else 8.0
        await asyncio.wait_for(self.room.local_participant.publish_track(self.track), timeout=timeout)
        if self.startup_trace:
            self.startup_trace.mark(StartupStage.TRACK_PUBLISHED)
        # Honesty first: the voice models may still be loading, and claiming
        # "speaking" over silence reads as a broken page. The room shows
        # "preparing" until the first sentence's audio actually exists.
        await self.send({"type": "state", "state": "preparing"})
        if self.startup_trace:
            self.startup_trace.mark(StartupStage.READY_ACKNOWLEDGED)

        segments = self.lecture.segments
        position = self.lecture.position

        # The whole lecture as a flat list, so we can always see the NEXT sentence.
        script: list[tuple[int, int, int, str]] = []  # (segment, sentence, slide, text)
        for s_index, segment in enumerate(segments):
            for t_index, sentence in enumerate(split_sentences(segment["text"])):
                script.append((s_index, t_index, segment["slide"], sentence))

        # Kokoro renders at ~1.6x realtime — too slow to start a sentence on demand,
        # but plenty fast to have the NEXT one ready while this one is playing. The
        # lecture text is known in advance, so we simply stay one sentence ahead.
        index = 0
        upcoming: asyncio.Task[np.ndarray] | None = None
        current_slide = -1

        while index < len(script) and not self.closed:
            s_index, t_index, slide, sentence = script[index]

            if slide != current_slide:
                await self.send({"type": "slide", "n": slide})
                current_slide = slide

            try:
                if upcoming:
                    audio = await upcoming
                elif index == 0 and self.startup_trace:
                    audio = await asyncio.wait_for(
                        self.sentence_audio(s_index, t_index, sentence),
                        timeout=self.startup_trace.remaining(),
                    )
                else:
                    audio = await self.sentence_audio(s_index, t_index, sentence)
            except (asyncio.TimeoutError, TimeoutError):
                await self._startup_failed("first_audio_timeout", "The lecturer could not start audio in time. Please retry.")
                return
            upcoming = None

            if index == 0:
                if not len(audio):
                    await self._startup_failed("first_audio_unavailable", "The lecturer's voice is unavailable. Please retry.")
                    return
                # The first audio exists - NOW "speaking" is true.
                await self.send({"type": "state", "state": "lecturing"})

            # Have the next sentence ready before speaking this one.
            if index + 1 < len(script):
                next_s, next_t, _, next_text = script[index + 1]
                upcoming = asyncio.create_task(self.sentence_audio(next_s, next_t, next_text))

            finished = await self.play(audio)
            if index == 0 and not self._first_frame_recorded:
                await self._startup_failed("first_frame_unavailable", "The lecture audio stream did not start. Please retry.")
                return
            if self.closed:
                break
            if finished:
                index += 1

            # The student raised a hand: the sentence above was allowed to finish
            # (that is the whole point), and only now does the lecturer respond.
            if self.hand_raised.is_set():
                if upcoming:
                    upcoming.cancel()
                    upcoming = None
                await self.handle_hand()
                await self.send({"type": "state", "state": "lecturing"})
                current_slide = -1      # re-announce the slide after the detour

        position.segment = len(segments)
        await self.send({"type": "state", "state": "ended"})

    async def _startup_failed(self, reason: str, detail: str) -> None:
        await self.send({"type": "progress", "stage": "problem", "detail": detail})
        await self.send({"type": "state", "state": "ended"})
        if self.startup_trace:
            payload = self.startup_trace.payload(
                mode=self.startup_mode,
                outcome="failed",
                reason_code=reason,
            )
            print(json.dumps(payload, sort_keys=True), flush=True)
            trace_path = os.getenv("STARTUP_TRACE_PATH")
            if trace_path:
                append_trace(Path(trace_path), payload)

    async def _wait_for_unmute(self, seconds: float) -> bool:
        try:
            await asyncio.wait_for(self.mic_unmuted.wait(), timeout=seconds)
            return True
        except asyncio.TimeoutError:
            return False

    async def handle_hand(self) -> None:
        """The raise-hand protocol, exactly as specified:

        hand up -> finish the sentence -> ask by name -> wait 4 s for an unmute
        -> friendly reminder -> wait again -> either take the question, or lower
        the hand and pull the class back with the resume line.
        """
        self.hand_raised.clear()
        if self.question_turn.start() is None:
            return

        await self.send({"type": "state", "state": "asking"})
        await self._play_prompt("ask")
        await self.send({"type": "hand", "state": "acked"})

        unmuted = await self._wait_for_unmute(4.0)
        if not unmuted:
            await self._play_prompt("remind")
            unmuted = await self._wait_for_unmute(8.0)

        answered = False
        if unmuted:
            self.question_turn.listen()
            self.hand_active = True
            try:
                answered = await self.collect_and_answer()
            finally:
                self.hand_active = False

        if not answered and self.question_turn.state is not TurnState.CLOSED:
            await self.question_turn.close("no_speech" if unmuted else "unmute_timeout")

        await self.send({"type": "hand", "state": "lowered"})
        if not answered:
            # No question came. Lower the hand and catch the room's attention.
            await self._play_prompt("resume")

    async def collect_and_answer(self) -> bool:
        """Capture the question, let the student edit the transcript, answer it.
        Returns False when nothing was ultimately asked."""
        await self.send({"type": "state", "state": "listening"})

        while not self.question_turn.review_ready.is_set():
            reason = self.question_turn.endpoint_reason()
            if reason:
                await self.question_turn.finalize(reason)
                break
            await asyncio.sleep(0.05)
        if self.question_turn.state is not TurnState.REVIEW:
            await self.send({"type": "progress", "stage": "problem", "detail": "No complete question was captured. Please raise your hand and try again."})
            return False
        heard = self.question_turn.transcript or ""

        # Nothing is asked on the student's behalf. We show them what we heard and
        # they send it, edit it first, or throw it away.
        await self.send({"type": "state", "state": "review"})
        await self.send({"type": "transcript", "text": heard})
        log(f"[question-turn] stage=review turn_id={self.question_turn.turn_id}")

        try:
            question = await asyncio.wait_for(self.confirmed.get(), timeout=REVIEW_TIMEOUT_S)
        except asyncio.TimeoutError:
            print("[lecture] no confirmation - resuming the lecture")
            await self.send({"type": "transcript", "text": None})
            await self.question_turn.close("review_timeout")
            return False

        if not question.strip():          # they cancelled
            print("[lecture] question cancelled")
            return False

        # The question is in — the capture window's job is done. Closing it now
        # keeps the student's still-open mic (or its echo of OUR answer) from
        # feeding Whisper while the lecturer talks.
        self.hand_active = False

        await self.send({"type": "state", "state": "answering"})
        log(f"[question-turn] stage=answering turn_id={self.question_turn.turn_id}")

        async def on_progress(stage: str, detail: str) -> None:
            log(f"[qa] {stage}: {detail}" if detail else f"[qa] {stage}")
            await self.send({"type": "progress", "stage": stage, "detail": detail})

        # Forward session identity to qa so citations carry programme/course/plan
        # scope. Falls back to empty strings when session_meta was not available
        # (only possible in standalone mode where worker.py is not used).
        scope = self.session_meta.as_citation_scope() if self.session_meta else {}
        try:
            result = await within_budget(
                Stage.TOTAL,
                answer_question(
                    question, lecture_id=None, sid=self.lecture.sid, on_progress=on_progress,
                    programme_id=scope.get("programme_id", ""),
                    course_id=scope.get("course_id", ""),
                    plan_version=scope.get("plan_version"),
                    lecture_id_str=scope.get("lecture_id", ""),
                ),
            )
        except StageTimeout:
            fallback = choose_fallback("agent", "total_qa_timeout")
            await self.send(fallback.event())
            result = {"answer": TROUBLE, "pages": [], "citations": []}
        await self.send({"type": "progress", "stage": "speaking", "detail": ""})

        await self.send(
            {
                "type": "answer",
                "payload": {
                    "question": question,
                    "answer": result["answer"],
                    "pages": result["pages"],
                    "citations": result.get("citations", []),
                },
            }
        )

        # The answer itself is not interruptible: it is short by design. Render
        # the NEXT sentence while the current one plays — the silent render gap
        # between sentences was why "Speaking" felt like it never finished.
        sentences = split_sentences(result["answer"])
        total = len(sentences)
        log(f"[speak] answer has {total} sentences, {len(result['answer'])} chars")
        upcoming: asyncio.Task[np.ndarray] | None = (
            asyncio.create_task(self.render(sentences[0])) if sentences else None
        )
        for index in range(total):
            await self.send(
                {
                    "type": "progress",
                    "stage": "speaking",
                    "detail": f"sentence {index + 1} of {total}",
                }
            )
            waited = time.perf_counter()
            audio = await upcoming if upcoming else await self.render(sentences[index])
            waited = time.perf_counter() - waited
            upcoming = (
                asyncio.create_task(self.render(sentences[index + 1]))
                if index + 1 < total
                else None
            )
            played = time.perf_counter()
            await self.play(audio, interruptible=False)
            played = time.perf_counter() - played
            speech = len(audio) / self.sample_rate
            log(
                f"[speak] sentence {index + 1}/{total}: waited {waited:.2f}s on TTS, "
                f"{speech:.1f}s of speech played in {played:.2f}s"
            )
        await self.question_turn.close("completed")
        return True


async def listen(session: LectureSession, track: rtc.RemoteAudioTrack, model) -> None:
    """The Listener agent: VAD for barge-in, faster-whisper for the question.

    Everything heavy here runs in a thread. Loading or running Whisper on the
    event loop stalls the Lecturer's audio pump, which sounds exactly like the
    lecture dying twenty seconds in — because it does.
    """
    from collections import deque

    stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1)
    buffer: list[np.ndarray] = []
    # Pre-roll: the VAD only fires AFTER you have been talking for a moment, so
    # without this ring the first syllables were cut off — Whisper then heard a
    # clipped fragment and often returned nothing. That was the biggest reason
    # "the mic recording does not catch".
    preroll: deque[np.ndarray] = deque(maxlen=50)  # ~0.5 s at 10 ms frames
    speech_ms = 0.0
    silence_ms = 0.0
    capturing = False
    # Adaptive threshold: a fixed 0.02 RMS misses quiet microphones entirely.
    # Track the noise floor and trigger a few times above it instead.
    noise_floor = 0.004

    async def transcribe(audio: np.ndarray) -> str:
        resolved_model = await session.dependencies.stt() if session.dependencies else model
        if resolved_model is None:
            return ""
        def run_stt() -> str:
            segments, _info = resolved_model.transcribe(audio, language="en")
            return " ".join(seg.text.strip() for seg in segments).strip()
        try:
            return await within_budget(Stage.STT, asyncio.to_thread(run_stt))
        except StageTimeout:
            await session.send(choose_fallback("stt", "stt_timeout").event())
            return ""

    async def flush_segment() -> None:
        nonlocal buffer, capturing, speech_ms
        if not buffer or not session.question_turn.turn_id:
            return
        audio = np.concatenate(buffer)
        buffer, capturing, speech_ms = [], False, 0
        preroll.clear()
        session.question_turn.add_stt(session.question_turn.turn_id, transcribe(audio))

    session.flush_capture = flush_segment

    async for event in stream:
        frame = event.frame
        samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        frame_ms = len(samples) / 16000 * 1000

        rms = float(np.sqrt(np.mean(samples**2)))
        threshold = min(0.02, max(0.006, noise_floor * 3.5))
        loud = rms > threshold
        if not loud:
            noise_floor = 0.95 * noise_floor + 0.05 * rms

        if loud:
            speech_ms += frame_ms
            silence_ms = 0
        else:
            silence_ms += frame_ms

        # Only the raise-hand window records anything: outside it the student is
        # muted anyway, and stray noise must never derail the lecture. The same
        # gate holds while the LECTURER is speaking — an open mic next to
        # speakers would otherwise feed our own voice back into Whisper.
        if not session.hand_active or session.speaking:
            preroll.append(samples)
            buffer, capturing, speech_ms = [], False, 0
            continue

        if loud:
            session.question_turn.observe_speech()

        if not capturing:
            preroll.append(samples)

        if not capturing and speech_ms >= SPEECH_TRIGGER_MS:
            capturing = True
            buffer = list(preroll)      # include the syllables from BEFORE the trigger
            print(f"[listener] capturing (threshold {threshold:.3f})")

        if capturing:
            buffer.append(samples)

            if silence_ms >= SILENCE_END_MS:
                await flush_segment()

        reason = session.question_turn.endpoint_reason()
        if reason:
            if buffer:
                await flush_segment()
            await session.question_turn.finalize(reason)

        if not capturing and silence_ms > 1000:
            speech_ms = 0


def prewarm(proc: agents.JobProcess) -> None:
    """Initialize only cheap state required to accept a LiveKit job."""
    proc.userdata["tts"] = None
    proc.userdata["dependencies"] = LazyDependencies()
    proc.userdata["audio_cache"] = AudioCache()

    # Generic prompts contain no learner data. Personalized clips are loaded
    # per authenticated session from PromptCache below.
    proc.userdata["prompts"] = {}
    proc.userdata["prompts_rate"] = 24000
    proc.userdata["prompt_cache"] = PromptCache(Path(os.getenv("PROMPT_CACHE_DIR", str(Path(__file__).parent / ".prompt-cache"))))

    proc.userdata["stt"] = None


def _startup_mode() -> str:
    value = os.getenv("STARTUP_MODE", "warm").strip().lower()
    return value if value in {"cold", "warm"} else "warm"


def _write_startup_trace(trace: StartupTrace, *, outcome: str, reason: str | None = None) -> None:
    payload = trace.payload(mode=_startup_mode(), outcome=outcome, reason_code=reason)
    print(json.dumps(payload, sort_keys=True), flush=True)
    trace_path = os.getenv("STARTUP_TRACE_PATH")
    if trace_path:
        append_trace(Path(trace_path), payload)


async def _fail_job_startup(
    ctx: agents.JobContext,
    trace: StartupTrace,
    *,
    reason: str,
    detail: str,
) -> None:
    try:
        await ctx.room.local_participant.publish_data(
            json.dumps({"type": "progress", "stage": "problem", "detail": detail}).encode("utf-8"),
            reliable=True,
        )
    except Exception:
        pass
    _write_startup_trace(trace, outcome="failed", reason=reason)


async def entrypoint(ctx: agents.JobContext) -> None:
    trace = StartupTrace()
    try:
        await asyncio.wait_for(ctx.connect(), timeout=trace.remaining())
    except (asyncio.TimeoutError, TimeoutError, ConnectionError):
        _write_startup_trace(trace, outcome="failed", reason="room_connect_failed")
        return
    trace.mark(StartupStage.ROOM_CONNECTED)

    raw_meta = getattr(ctx.room, "metadata", None) or ""
    try:
        metadata_kind = json.loads(raw_meta).get("schema_name")
    except (ValueError, AttributeError):
        metadata_kind = None
    if metadata_kind == "univai.section-session-meta":
        await _run_section(ctx, raw_meta, trace)
        return

    # Room names are lecture-<studentId>-week-N (the app's token route mints
    # them). rpartition on "-week-" is safe even though studentId itself
    # contains dashes (S-2026-000042).
    from protocol import parse_room_name

    sid, week = parse_room_name(ctx.room.name)

    # Parse extended session metadata from roomMetadata (App → Live contract).
    try:
        session_meta = LectureSessionMeta.from_room_metadata(
            ctx.room.name, raw_meta, sid=sid
        )
        if session_meta.week != week:
            raise SessionMetadataError(
                "roomMetadata.week does not match the LiveKit room name.",
                field="week",
            )
        trace.mark(StartupStage.METADATA_VALID)
        print(
            f"[lecture] session metadata: programme={session_meta.programme_id} "
            f"course={session_meta.course_id} plan={session_meta.plan_version}"
        )
    except SessionMetadataError as exc:
        print(f"[lecture] ERROR: {exc} (field={exc.field!r})")
        await _fail_job_startup(
            ctx,
            trace,
            reason="metadata_invalid",
            detail=f"Invalid lecture session metadata ({exc.field}): {exc}",
        )
        return

    try:
        lecture = await asyncio.wait_for(
            asyncio.to_thread(Lecture.load, sid, week, session_meta.artifact_id),
            timeout=trace.remaining(),
        )
        trace.mark(StartupStage.ARTIFACT_LOADED)
    except (OSError, ValueError, asyncio.TimeoutError, TimeoutError) as exc:
        await _fail_job_startup(
            ctx,
            trace,
            reason="artifact_unavailable",
            detail=f"The lecture artifact is unavailable: {exc}",
        )
        return
    print(f"[lecture] {sid} week {week}: {lecture.title} ({len(lecture.segments)} segments)")

    startup_mode = _startup_mode()
    session = LectureSession(
        ctx.room, lecture, ctx.proc.userdata["tts"], session_meta,
        startup_trace=trace, startup_mode=startup_mode,
        dependencies=ctx.proc.userdata["dependencies"],
        audio_cache=ctx.proc.userdata["audio_cache"],
    )
    prompts_rate = ctx.proc.userdata.get("prompts_rate")
    prompt_bank = ctx.proc.userdata["prompts"]
    try:
        display_name = session_meta.display_name
        engine = ctx.proc.userdata.get("tts")
        if isinstance(display_name, str):
            prompt_bank, cached_rate = ctx.proc.userdata["prompt_cache"].load(
                learner_id=sid, display_name=display_name, language="en",
                voice=str(getattr(engine, "voice", os.getenv("KOKORO_VOICE", "default"))),
                model=str(getattr(engine, "name", os.getenv("TTS_ENGINE", "kokoro"))),
                model_version=str(getattr(engine, "model_version", os.getenv("TTS_MODEL_VERSION", "1"))),
                generic=prompt_bank,
            )
            prompts_rate = cached_rate or prompts_rate
    except (ValueError, TypeError):
        prompt_bank = ctx.proc.userdata["prompts"]
    session.prompts = {
        key: session._fit(audio, prompts_rate)
        for key, audio in prompt_bank.items()
    }
    stt_model = ctx.proc.userdata["stt"]

    @ctx.room.on("track_subscribed")
    def on_track(track: rtc.Track, *_: object) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(listen(session, track, stt_model))

    @ctx.room.on("data_received")
    def on_data(packet: rtc.DataPacket) -> None:
        # The student pressed Send (possibly after editing) or Cancel.
        participant = getattr(packet, "participant", None)
        if participant is not None and getattr(participant, "identity", sid) != sid:
            return
        try:
            message = json.loads(packet.data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return
        if message.get("type") == "question":
            confirmed = session.question_turn.confirm(str(message.get("text", "")))
            if confirmed is not None:
                session.confirmed.put_nowait(confirmed)
        elif message.get("type") == "cancel":
            async def cancel_review() -> None:
                if await session.question_turn.cancel():
                    session.confirmed.put_nowait("")
            asyncio.create_task(cancel_review())
        elif message.get("type") == "raise_hand":
            if session.question_turn.state not in {TurnState.IDLE, TurnState.CLOSED}:
                session.question_turn.start()
                return
            log("[lecture] hand raised")
            session.hand_raised.set()
        elif message.get("type") == "mic":
            if message.get("muted"):
                session.mic_unmuted.clear()
                session.question_turn.request_mute()
                turn_id = session.question_turn.turn_id
                async def finalize_after_drain() -> None:
                    await asyncio.sleep(session.question_turn.config.mute_drain_ms / 1000)
                    if turn_id != session.question_turn.turn_id or session.question_turn.state is not TurnState.LISTENING:
                        return
                    if session.flush_capture:
                        await session.flush_capture()
                    reason = session.question_turn.endpoint_reason()
                    if reason == "mic_muted":
                        await session.question_turn.finalize(reason)
                asyncio.create_task(finalize_after_drain())
            else:
                session.mic_unmuted.set()
        elif message.get("type") == "startup_audio_playing":
            elapsed = message.get("client_elapsed_ms")
            if isinstance(elapsed, (int, float)) and 0 <= elapsed <= 60_000:
                client_event = {
                    "schema": "univai.live.client-first-audio",
                    "trace_id": trace.trace_id,
                    "mode": session.startup_mode,
                    "client_elapsed_ms": round(elapsed),
                }
                print(json.dumps(client_event, sort_keys=True), flush=True)
                trace_path = os.getenv("STARTUP_TRACE_PATH")
                if trace_path:
                    append_trace(Path(trace_path), client_event)

    @ctx.room.on("disconnected")
    def on_disconnected(*_: object) -> None:
        asyncio.create_task(session.question_turn.close("disconnect"))

    await session.run()


async def _run_section(ctx: agents.JobContext, raw_meta: str, trace: StartupTrace) -> None:
    """Minimal LiveKit adapter around the repository-local section controller."""
    from protocols.section_session import (
        SECTION_REFERENCE_VERSION,
        SectionContractError,
        SectionSessionMetaV1,
        SectionSessionReferenceV2,
    )
    from section_controller import SectionController

    prefix, separator, _week = ctx.room.name.rpartition("-week-")
    learner_id = prefix[len("section-") :] if separator and prefix.startswith("section-") else ""
    try:
        decoded = json.loads(raw_meta)
        if isinstance(decoded, dict) and decoded.get("schema_version") == SECTION_REFERENCE_VERSION:
            reference = SectionSessionReferenceV2.from_room_metadata(
                raw_meta,
                authenticated_learner_id=learner_id,
            )
            from common.db import fetch_one

            row = await asyncio.wait_for(
                asyncio.to_thread(
                    fetch_one,
                    # A section opens on the clock, like the week's quiz. The
                    # 45-120 clamp with a 60-minute legacy default mirrors the
                    # App's scriptDurationMinutes, so both gates agree to the
                    # minute on when a lecture ended.
                    """SELECT sp.programme_id, p.name AS programme_title,
                              sp.course_id, sp.week, sp.lecture_id,
                              sp.approved_plan_version, sp.pack_payload,
                              (now() + ((SELECT offset_ms FROM clock_state WHERE id = 1)
                                        || ' milliseconds')::interval)
                                >= l.starts_at
                                   + (CASE WHEN (la.script_payload->>'durationMinutes') ~ '^[0-9]+$'
                                            AND (la.script_payload->>'durationMinutes')::int
                                                BETWEEN 45 AND 120
                                           THEN (la.script_payload->>'durationMinutes')::int
                                           ELSE 60 END || ' minutes')::interval
                                AS lecture_ended
                         FROM section_packs sp
                         JOIN programmes p ON p.id::text = sp.programme_id
                          AND p.student_id = sp.tenant_id
                          AND p.status = 'approved'
                          AND p.plan_version = sp.approved_plan_version
                         JOIN lectures l ON l.student_id = sp.tenant_id
                          AND l.week = sp.week
                         LEFT JOIN lecture_artifacts la
                           ON la.artifact_id = l.lecture_artifact_id
                        WHERE sp.tenant_id = %s AND sp.section_pack_id = %s
                        LIMIT 1""",
                    (learner_id, reference.section_pack_id),
                ),
                timeout=trace.remaining(),
            )
            if not row:
                raise SectionContractError("section pack was not found", field="section_pack_id")
            meta = SectionSessionMetaV1.from_storage(
                reference,
                programme_id=str(row["programme_id"]),
                programme_title=str(row["programme_title"]),
                course_id=str(row["course_id"]),
                week=int(row["week"]),
                lecture_id=str(row["lecture_id"]),
                plan_version=int(row["approved_plan_version"]),
                lecture_ended=bool(row["lecture_ended"]),
                pack=row["pack_payload"],
            )
        else:
            # One-version rolling-deploy compatibility. New App releases send
            # only the v2 reference; legacy metadata remains accepted briefly.
            meta = SectionSessionMetaV1.from_room_metadata(
                raw_meta,
                authenticated_learner_id=learner_id,
            )
        trace.mark(StartupStage.METADATA_VALID)
        trace.mark(StartupStage.ARTIFACT_LOADED)
    except (SectionContractError, asyncio.TimeoutError, TimeoutError, ValueError) as exc:
        await ctx.room.local_participant.publish_data(json.dumps({
            "type": "section_error", "schema_version": "1.0.0",
            "payload": {"field": getattr(exc, "field", "section_pack_id"), "message": str(exc)},
        }).encode(), reliable=True)
        _write_startup_trace(trace, outcome="failed", reason="section_artifact_unavailable")
        return

    engine = ctx.proc.userdata.get("tts")
    sample_rate = engine.sample_rate if engine else 24000
    source = rtc.AudioSource(sample_rate, 1)
    track = rtc.LocalAudioTrack.create_audio_track("section-guide", source)
    await asyncio.wait_for(ctx.room.local_participant.publish_track(track), timeout=trace.remaining())
    trace.mark(StartupStage.TRACK_PUBLISHED)
    await ctx.room.local_participant.publish_data(json.dumps({"type": "section_ready", "schema_version": "1.0.0"}).encode(), reliable=True)
    trace.mark(StartupStage.READY_ACKNOWLEDGED)
    first_content = False

    async def emit(event: dict) -> None:
        nonlocal first_content
        await ctx.room.local_participant.publish_data(json.dumps(event).encode(), reliable=True)
        if not first_content:
            first_content = True
            trace.mark(StartupStage.FIRST_FRAME)
            payload = trace.payload(mode=_startup_mode())
            print(json.dumps(payload, sort_keys=True), flush=True)
            trace_path = os.getenv("STARTUP_TRACE_PATH")
            if trace_path:
                append_trace(Path(trace_path), payload)
            asyncio.create_task(ctx.proc.userdata["dependencies"].warm())

    async def speak(text: str) -> None:
        nonlocal engine
        if engine is None:
            try:
                engine = await ctx.proc.userdata["dependencies"].tts()
            except Exception:
                await emit(choose_fallback("tts", "tts_unavailable").event())
                return
        try:
            audio = await within_budget(Stage.TTS, asyncio.to_thread(engine.render, text))
        except StageTimeout:
            await emit(choose_fallback("tts", "tts_timeout").event())
            return
        engine_rate = getattr(engine, "sample_rate", sample_rate)
        if engine_rate != sample_rate and len(audio):
            output_length = int(len(audio) * sample_rate / engine_rate)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, output_length),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
        for start in range(0, len(audio), sample_rate // 10):
            pcm = (np.clip(audio[start : start + sample_rate // 10], -1, 1) * 32767).astype(np.int16)
            await source.capture_frame(rtc.AudioFrame(data=pcm.tobytes(), sample_rate=sample_rate, num_channels=1, samples_per_channel=len(pcm)))

    controller = SectionController(meta, emit, speak)
    commands: asyncio.Queue[dict] = asyncio.Queue()

    @ctx.room.on("data_received")
    def on_section_data(packet: rtc.DataPacket) -> None:
        participant = getattr(packet, "participant", None)
        if participant is None or getattr(participant, "identity", "") != meta.learner_id:
            return
        try:
            value = json.loads(packet.data.decode())
        except (ValueError, UnicodeDecodeError):
            return
        if value.get("type") in {"section_submit", "todo_ack", "section_complete", "raise_hand"}:
            commands.put_nowait(value)

    await controller.start()
    while not controller.completed:
        command = await commands.get()
        if command["type"] == "section_submit":
            await controller.submit(str(command.get("submission_id", "")), activity_index=int(command.get("activity_index", -1)), text=str(command.get("text", "")))
        elif command["type"] == "todo_ack":
            await controller.acknowledge_todo(int(command.get("todo_index", -1)))
        elif command["type"] == "raise_hand":
            checkpoint = await controller.interrupt()
            await controller.resume(checkpoint)
        else:
            await controller.complete()


server = agents.AgentServer(initialize_process_timeout=10.0)
server.setup_fnc = prewarm
# Keep this worker unnamed so rooms receive automatic dispatch. Naming an agent
# opts into explicit dispatch, which the UnivAI token routes do not request.
server.rtc_session(entrypoint)


if __name__ == "__main__":
    agents.cli.run_app(server)
