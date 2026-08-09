# UnivAI Live — mouth + ears

The realtime voice cave of **UnivAI ("Jamieh")**: LiveKit rooms, TTS, STT —
the lecturer that actually speaks, listens, and answers raised hands.

## Standalone simulator

The default simulator needs only Python 3.12+. It uses the same App-visible
message vocabulary, a project-authored lecture, scripted STT, deterministic
grounded Q&A, and silent audio. It does not contact LiveKit, PostgreSQL, Agent,
an LLM, or any TTS/STT model.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe simulate.py run
.\.venv\Scripts\python.exe simulate.py smoke
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On Linux replace the executable with `.venv/bin/python`. The simulator sets
standalone mode explicitly. You can also use `UNIVAI_MODE=standalone`.
Standalone fixture providers are rejected when `UNIVAI_ENV=production`.

The fixture participant is `S-2026-000042`; its room convention is
`lecture-S-2026-000042-week-1`. The simulator exercises raise hand, mic,
question, transcript review, answer, and completion without a microphone or
camera. Voice quality and real network timing are deliberately not tested.

## Integrated LiveKit worker

```bash
# from the UnivAI campus root (this repo checked out as a submodule there)
.venv/Scripts/python.exe UnivAI-live/worker.py dev
```

Registers against the LiveKit server in `.env` (local docker `:7880` by
default) and waits for lecture rooms. Logs land in the campus `logs/`.
Keep `UNIVAI_MODE=integrated` (the default). Missing LiveKit, shared services,
or model dependencies fail instead of activating fixtures.

Production uses the current Agents `AgentServer` API and must be built from the
campus root with `docker build -f UnivAI-live/Dockerfile .`. It starts the real
integrated worker in `start` mode; build `Dockerfile.simulator` explicitly when
you want the standalone simulator.

## What's inside

| File | Job |
|---|---|
| `worker.py` | the live class: joins lecture and section rooms, loads artifacts from PostgreSQL, synthesizes narration, drives slides, and runs the raise-hand protocol |
| `qa.py` | a raised-hand question → RAG (the Brain, over MCP) → short spoken answer with page citations |
| `tts.py` | the voices: Kokoro (rich) and Piper (~10x realtime), synthesized on demand |
| `models/` | the voice model binaries (Kokoro, Piper) — gitignored, fetched by `make models` from the campus root |
| `stt-project/` | speech-to-text / XTTS research prototypes |

## How it connects

- Reads the **campus root `.env`** (LiveKit keys, model paths, student name)
- Imports the shared plumbing from the campus `services/common/`
  (clock, db, LLM adapter, RAG client)
- Loads lecture scripts and section packs from PostgreSQL; no learner lecture folders are read or written
- STT is faster-whisper; the mic is only *heard* inside the raise-hand window

## Protocol and safety

`protocol.py` is the canonical validation source. Live sends `slide`, `state`,
`answer`, `transcript`, `progress`, and `hand`; the App sends `raise_hand`,
`mic`, `question`, and `cancel`. States are `connecting`, `preparing`,
`lecturing`, `asking`, `listening`, `review`, `answering`, and `ended`.
Questions are limited to 500 characters in the simulator and unknown messages
are rejected without entering the controller.

`stt-project/` remains an optional GPU research prototype. Real TTS/STT may
download large models and must be tested explicitly through the main
integrated environment.

This directory is a Git submodule. Merge Live changes in its own repository
first, then update the main UnivAI gitlink. Local submodule file changes are
not automatically included in a main-repository commit.
## Raised-hand endpointing

Question capture uses Silero speech probability, keeps the 800 ms pause as an STT segment boundary, and joins every turn-scoped segment before review. A turn ends after 2,500 ms final silence or 300 ms after mute, with 30 s to first speech and a 45 s duration bound. If recognition times out or returns no text, the existing review box opens empty so the learner can type instead of losing the turn. The worker attaches both newly subscribed and already-published microphone tracks, which removes the connect/publish race. Override only with validated `QUESTION_SEGMENT_MS`, `QUESTION_FINAL_SILENCE_MS`, `QUESTION_MUTE_DRAIN_MS`, `QUESTION_FIRST_SPEECH_MS`, and `QUESTION_MAX_DURATION_MS`; invalid bounds fail startup.

Real acceptance requires a configured LiveKit room and microphone: ask a covered multi-pause question ten times, cancel a fragmented turn before asking another, and verify one complete editable transcript and exactly one grounded retrieval per confirmed turn. Unit tests use fake clocks and are not evidence of microphone behavior.
