# UnivAI Live — mouth + ears

The realtime voice cave of **UnivAI ("Jamieh")**: LiveKit rooms, TTS, STT —
the lecturer that actually speaks, listens, and answers raised hands.

## Run it

```bash
# from the UnivAI campus root (this repo checked out as a submodule there)
.venv/Scripts/python.exe UnivAI-live/worker.py dev
```

Registers against the LiveKit server in `.env` (local docker `:7880` by
default) and waits for lecture rooms. Logs land in the campus `logs/`.

## What's inside

| File | Job |
|---|---|
| `worker.py` | the live class: joins `lecture-week-N` rooms, plays the pre-recorded lecture, drives the slides, runs the raise-hand protocol |
| `qa.py` | a raised-hand question → RAG (the Brain, over MCP) → short spoken answer with page citations |
| `tts.py` | the voices: Kokoro (rich, used for pre-rendering) and Piper (~10x realtime, used live) |
| `prerender_audio.py` | records the whole lecture + the personalized raise-hand prompts to disk, so lectures start instantly |
| `stt-project/` | speech-to-text / XTTS research prototypes |

## How it connects

- Reads the **campus root `.env`** (LiveKit keys, model paths, student name)
- Imports the shared plumbing from the campus `services/common/`
  (clock, db, LLM adapter, RAG client)
- Plays pre-rendered audio from the campus `lectures/week-N/audio/`
- STT is faster-whisper; the mic is only *heard* inside the raise-hand window
