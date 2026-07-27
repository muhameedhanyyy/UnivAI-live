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

## What's inside

| File | Job |
|---|---|
| `worker.py` | the live class: joins `lecture-week-N` rooms, plays the pre-recorded lecture, drives the slides, runs the raise-hand protocol |
| `qa.py` | a raised-hand question → RAG (the Brain, over MCP) → short spoken answer with page citations |
| `tts.py` | the voices: Kokoro (rich, used for pre-rendering) and Piper (~10x realtime, used live) |
| `prerender_audio.py` | records the whole lecture + the personalized raise-hand prompts to disk, so lectures start instantly |
| `models/` | the voice model binaries (Kokoro, Piper) — gitignored, fetched by `make models` from the campus root |
| `stt-project/` | speech-to-text / XTTS research prototypes |

## How it connects

- Reads the **campus root `.env`** (LiveKit keys, model paths, student name)
- Imports the shared plumbing from the campus `services/common/`
  (clock, db, LLM adapter, RAG client)
- Plays pre-rendered audio from the campus `lectures/week-N/audio/`
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
