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
import sys
from pathlib import Path

import numpy as np

# Lecture titles land in log prints; a redirected Windows stdout is cp1252.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services"))  # campus plumbing
sys.path.insert(0, str(Path(__file__).resolve().parent))                   # tts.py, same cave

from common.sentences import split_sentences  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LECTURES_DIR = ROOT / "lectures"


def prerender_all(sid: str | None = None, log=print) -> dict:
    from tts import load_engine  # imports onnx models — keep at call time

    engine = load_engine()
    rendered = 0

    # Multi-tenant: render this student's course under lectures/<sid>/week-N/.
    # Without a sid, fall back to the legacy global lectures/week-N/.
    base = LECTURES_DIR / sid if sid else LECTURES_DIR
    for folder in sorted(base.glob("week-*")):
        script_path = folder / "script.json"
        if not script_path.exists():
            continue
        script = json.loads(script_path.read_text("utf-8"))
        audio_dir = folder / "audio"
        audio_dir.mkdir(exist_ok=True)

        log(f"[prerender] {folder.name}: {script['title']}")
        for s_index, segment in enumerate(script["segments"]):
            for t_index, sentence in enumerate(split_sentences(segment["text"])):
                target = audio_dir / f"s{s_index}-t{t_index}.npy"
                np.save(target, engine.render(sentence).astype(np.float32))
                rendered += 1

        (audio_dir / "meta.json").write_text(
            json.dumps({"sample_rate": engine.sample_rate}), encoding="utf-8"
        )

    log(f"[prerender] done: {rendered} clips at {engine.sample_rate} Hz")
    return {"ok": True, "clips": rendered, "sample_rate": engine.sample_rate}


if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        print(json.dumps(prerender_all(sid)))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        raise SystemExit(1)
