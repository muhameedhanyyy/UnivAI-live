"""The live lecture: one worker process, two agent identities in the LiveKit room.

    Lecturer  publishes the TTS audio track and drives the slides
    Listener   subscribes to the student's mic, runs VAD + STT, and interrupts

State machine (exactly as specified):

    LECTURING    stream script.json through TTS, sentence by sentence.
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
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# This cave (UnivAI-live) is checked out inside the UnivAI campus repo; the
# shared plumbing (clock, db, LLM, RAG client) lives there in services/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services"))

import numpy as np
from dotenv import load_dotenv
from livekit import agents, rtc

from common.device import whisper_settings, describe  # noqa: E402
from common.sentences import split_sentences  # noqa: E402
from qa import answer_question  # noqa: E402
from tts import load_live_engine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# "localhost" resolves to ::1 first on Windows, and the Rust rtc client times
# out on a dropped ::1 before falling back to IPv4 — dial v4 loopback directly.
_url = os.getenv("LIVEKIT_URL", "")
if "://localhost" in _url:
    os.environ["LIVEKIT_URL"] = _url.replace("://localhost", "://127.0.0.1")

LECTURES_DIR = ROOT / "lectures"
STT_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "base")

SPEECH_TRIGGER_MS = 300     # this much speech from the student = a barge-in
SILENCE_END_MS = 800        # this much silence = they have finished asking
REVIEW_TIMEOUT_S = 120      # how long we hold the lecture while they edit the transcript


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
    position: Position = field(default_factory=Position)
    # Pre-rendered voice (prerender_audio.py, this cave). When present, the
    # lecture NEVER touches a TTS model — it plays from disk.
    audio_dir: Path | None = None
    audio_rate: int | None = None

    @staticmethod
    def load(week: int) -> "Lecture":
        folder = LECTURES_DIR / f"week-{week}"
        script = json.loads((folder / "script.json").read_text("utf-8"))
        lecture = Lecture(week=week, title=script["title"], segments=script["segments"])
        meta = folder / "audio" / "meta.json"
        if meta.exists():
            lecture.audio_dir = folder / "audio"
            lecture.audio_rate = int(json.loads(meta.read_text("utf-8"))["sample_rate"])
        return lecture

    def clip(self, segment: int, sentence: int) -> np.ndarray | None:
        if not self.audio_dir:
            return None
        path = self.audio_dir / f"s{segment}-t{sentence}.npy"
        return np.load(path) if path.exists() else None


# ---------------------------------------------------------------- the worker


class LectureSession:
    def __init__(self, room: rtc.Room, lecture: Lecture, tts) -> None:
        self.room = room
        self.lecture = lecture
        # Loaded in prewarm(). May be None on a RAM-starved machine — the
        # lecture still plays (pre-rendered on disk); only live answers then
        # need the on-demand Piper fallback in _engine().
        self.tts = tts
        self._engine_retry = False

        # Engines differ: Piper is 22.05 kHz, XTTS is 24 kHz. Publishing at the
        # wrong rate does not fail — it just makes the lecturer sound wrong.
        # The pre-rendered bank's rate wins: that is most of what gets played.
        self.sample_rate = lecture.audio_rate or (tts.sample_rate if tts else 24000)
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
                from tts import PiperTTS

                self.tts = await asyncio.to_thread(PiperTTS)
                print("[tts] Piper loaded on demand for live speech")
            except Exception as exc:
                print(f"[tts] no live engine available: {exc}")
        return self.tts

    async def render(self, text: str) -> np.ndarray:
        """Synthesis is CPU-bound, so keep it off the event loop."""
        engine = await self._engine()
        if engine is None:
            # Nothing can speak. The text still reaches the browser as data.
            return np.zeros(0, dtype=np.float32)
        audio = await asyncio.to_thread(engine.render, text)
        return self._fit(audio, engine.sample_rate)

    async def sentence_audio(self, segment: int, sentence: int, text: str) -> np.ndarray:
        """Disk first: a pre-rendered sentence costs a read, not a model."""
        clip = self.lecture.clip(segment, sentence)
        if clip is not None:
            return self._fit(clip, self.lecture.audio_rate)
        return await self.render(text)

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

    # -- the state machine ------------------------------------------------------

    async def run(self) -> None:
        await self.room.local_participant.publish_track(self.track)
        # Honesty first: the voice models may still be loading, and claiming
        # "speaking" over silence reads as a broken page. The room shows
        # "preparing" until the first sentence's audio actually exists.
        await self.send({"type": "state", "state": "preparing"})

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

            audio = (
                await upcoming
                if upcoming
                else await self.sentence_audio(s_index, t_index, sentence)
            )
            upcoming = None

            if index == 0:
                # The first audio exists - NOW "speaking" is true.
                await self.send({"type": "state", "state": "lecturing"})

            # Have the next sentence ready before speaking this one.
            if index + 1 < len(script):
                next_s, next_t, _, next_text = script[index + 1]
                upcoming = asyncio.create_task(self.sentence_audio(next_s, next_t, next_text))

            finished = await self.play(audio)
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

        await self.send({"type": "state", "state": "asking"})
        if "ask" in self.prompts:
            await self.play(self.prompts["ask"], interruptible=False)
        await self.send({"type": "hand", "state": "acked"})

        unmuted = await self._wait_for_unmute(4.0)
        if not unmuted:
            if "remind" in self.prompts:
                await self.play(self.prompts["remind"], interruptible=False)
            unmuted = await self._wait_for_unmute(8.0)

        answered = False
        if unmuted:
            self.hand_active = True
            try:
                answered = await self.collect_and_answer()
            finally:
                self.hand_active = False

        await self.send({"type": "hand", "state": "lowered"})
        if not answered and "resume" in self.prompts:
            # No question came. Lower the hand and catch the room's attention.
            await self.play(self.prompts["resume"], interruptible=False)

    async def collect_and_answer(self) -> bool:
        """Capture the question, let the student edit the transcript, answer it.
        Returns False when nothing was ultimately asked."""
        await self.send({"type": "state", "state": "listening"})

        try:
            heard = await asyncio.wait_for(self.heard.get(), timeout=20)
        except asyncio.TimeoutError:
            # They unmuted but never actually asked anything. Carry on.
            return False

        # Nothing is asked on the student's behalf. We show them what we heard and
        # they send it, edit it first, or throw it away.
        await self.send({"type": "state", "state": "review"})
        await self.send({"type": "transcript", "text": heard})
        log(f"[lecture] heard: {heard!r} - waiting for the student to confirm")

        try:
            question = await asyncio.wait_for(self.confirmed.get(), timeout=REVIEW_TIMEOUT_S)
        except asyncio.TimeoutError:
            print("[lecture] no confirmation - resuming the lecture")
            await self.send({"type": "transcript", "text": None})
            return False

        if not question.strip():          # they cancelled
            print("[lecture] question cancelled")
            return False

        # The question is in — the capture window's job is done. Closing it now
        # keeps the student's still-open mic (or its echo of OUR answer) from
        # feeding Whisper while the lecturer talks.
        self.hand_active = False

        await self.send({"type": "state", "state": "answering"})
        log(f"[lecture] question: {question}")

        async def on_progress(stage: str, detail: str) -> None:
            log(f"[qa] {stage}: {detail}" if detail else f"[qa] {stage}")
            await self.send({"type": "progress", "stage": stage, "detail": detail})

        result = await answer_question(question, lecture_id=None, on_progress=on_progress)
        await self.send({"type": "progress", "stage": "speaking", "detail": ""})

        await self.send(
            {
                "type": "answer",
                "payload": {
                    "question": question,
                    "answer": result["answer"],
                    "pages": result["pages"],
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

        if not capturing:
            preroll.append(samples)

        if not capturing and speech_ms >= SPEECH_TRIGGER_MS:
            capturing = True
            buffer = list(preroll)      # include the syllables from BEFORE the trigger
            print(f"[listener] capturing (threshold {threshold:.3f})")

        if capturing:
            buffer.append(samples)

            if silence_ms >= SILENCE_END_MS:
                audio = np.concatenate(buffer) if buffer else np.zeros(1, dtype=np.float32)
                buffer, capturing, speech_ms, silence_ms = [], False, 0, 0
                preroll.clear()

                def run_stt(samples: np.ndarray) -> str:
                    segments, _info = model.transcribe(samples, language="en")
                    return " ".join(seg.text.strip() for seg in segments).strip()

                stt_started = time.perf_counter()
                text = await asyncio.to_thread(run_stt, audio)
                log(
                    f"[listener] STT of {len(audio) / 16000:.1f}s audio took "
                    f"{time.perf_counter() - stt_started:.2f}s"
                )
                if text:
                    await session.heard.put(text)

        if not capturing and silence_ms > 1000:
            speech_ms = 0


def prewarm(proc: agents.JobProcess) -> None:
    """Load the models once, when the worker starts — not when a student joins.

    Every piece here is allowed to fail: lectures play from disk, prompts come
    pre-rendered from disk, and a missing engine only degrades live answers."""
    try:
        engine = load_live_engine()
    except Exception as exc:
        print(f"[tts] engine failed to load ({exc}) - lecture audio comes from disk")
        engine = None
    proc.userdata["tts"] = engine

    # The raise-hand prompts, personalized, SAME voice as the lecture. Disk
    # first (prerender_audio.py wrote them); render only if they are missing.
    prompts: dict[str, np.ndarray] = {}
    rate = engine.sample_rate if engine else 24000
    prompts_dir = LECTURES_DIR / "_prompts"
    meta = prompts_dir / "meta.json"
    if meta.exists():
        rate = int(json.loads(meta.read_text("utf-8"))["sample_rate"])
        for key in ("ask", "remind", "resume"):
            path = prompts_dir / f"{key}.npy"
            if path.exists():
                prompts[key] = np.load(path)
        print(f"[lecture] raise-hand prompts loaded from disk ({len(prompts)})")
    elif engine is not None:
        student = os.getenv("STUDENT_NAME", "there")
        prompt_texts = {
            "ask": f"Yes, {student}? Do you have a question? Unmute your microphone and go ahead.",
            "remind": f"{student}, your hand is still raised. Unmute whenever you are ready, I am listening.",
            "resume": "No question? No problem. Alright everyone, eyes back on the slides, and let us continue!",
        }
        prompts = {key: engine.render(text) for key, text in prompt_texts.items()}
        print(f"[lecture] raise-hand prompts rendered for '{student}'")
    proc.userdata["prompts"] = prompts
    proc.userdata["prompts_rate"] = rate

    from faster_whisper import WhisperModel

    stt_device, compute_type = whisper_settings()
    try:
        model = WhisperModel(STT_MODEL_SIZE, device=stt_device, compute_type=compute_type)
        print(f"[listener] faster-whisper '{STT_MODEL_SIZE}' on {describe()} ({compute_type})")
    except Exception as exc:
        # CUDA whisper needs cuDNN; fall back rather than leave the room deaf.
        print(f"[listener] {stt_device} unavailable ({exc}); using CPU")
        model = WhisperModel(STT_MODEL_SIZE, device="cpu", compute_type="int8")
    proc.userdata["stt"] = model


async def entrypoint(ctx: agents.JobContext) -> None:
    await ctx.connect()

    # Room names are lecture-week-N, so the week is the source of truth here.
    week = int(ctx.room.name.rsplit("-", 1)[-1])
    lecture = Lecture.load(week)
    print(f"[lecture] week {week}: {lecture.title} ({len(lecture.segments)} segments)")

    session = LectureSession(ctx.room, lecture, ctx.proc.userdata["tts"])
    prompts_rate = ctx.proc.userdata.get("prompts_rate")
    session.prompts = {
        key: session._fit(audio, prompts_rate)
        for key, audio in ctx.proc.userdata["prompts"].items()
    }
    stt_model = ctx.proc.userdata["stt"]

    @ctx.room.on("track_subscribed")
    def on_track(track: rtc.Track, *_: object) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(listen(session, track, stt_model))

    @ctx.room.on("data_received")
    def on_data(packet: rtc.DataPacket) -> None:
        # The student pressed Send (possibly after editing) or Cancel.
        try:
            message = json.loads(packet.data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return
        if message.get("type") == "question":
            session.confirmed.put_nowait(str(message.get("text", "")))
        elif message.get("type") == "cancel":
            session.confirmed.put_nowait("")
        elif message.get("type") == "raise_hand":
            log("[lecture] hand raised")
            session.hand_raised.set()
        elif message.get("type") == "mic":
            if message.get("muted"):
                session.mic_unmuted.clear()
            else:
                session.mic_unmuted.set()

    await session.run()


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            # Prewarm loads Kokoro + Whisper and renders the raise-hand prompts;
            # the SDK's default 10 s initialize timeout kills the process mid-load.
            initialize_process_timeout=180.0,
        )
    )
