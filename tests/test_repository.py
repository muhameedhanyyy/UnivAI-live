from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_live_modules_parse_without_loading_external_models() -> None:
    for name in (
        "worker.py",
        "qa.py",
        "tts.py",
        "startup.py",
        "audio_cache.py",
        "prerender_audio.py",
    ):
        path = ROOT / name
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
