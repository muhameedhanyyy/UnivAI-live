"""Local TTS for the Lecturer agent.

Measured on this project's dev machine (RTX 3060 laptop, 6 GB), warm:

    XTTS-v2 (Coqui)   0.55x realtime   slower than speech - the lecture stalls
    Kokoro-82M         1.6x realtime   best voice; needs a sentence of lead time
    Piper (lessac)      11x realtime   instant, plainer voice

Kokoro is the default: it sounds like a person, and 1.6x is fast enough *because
the lecture script is known in advance* - the worker renders the next sentence
while the current one is playing (see worker.py), so the lead time is hidden.

Piper is the fallback for anything that must be instant, and for machines where
Kokoro cannot keep up. The latency gate below reports the real measurement
rather than letting a demo fail quietly.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from campus_imports import configure_campus_imports

configure_campus_imports()
from common.device import cuda_available, device, describe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# XTTS asks you to accept its licence on stdin. A background worker has no stdin,
# so without this it hangs forever with no output.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

TTS_ENGINE = os.getenv("TTS_ENGINE", "kokoro").lower()
# Live speech has a student waiting for each sentence, so it can use a separate
# low-latency engine choice from TTS_ENGINE.
TTS_LIVE_ENGINE = os.getenv("TTS_LIVE_ENGINE", "").lower()
LATENCY_GATE_S = float(os.getenv("TTS_LATENCY_GATE_S", "4.0"))

PIPER_MODEL = os.getenv("PIPER_MODEL", "UnivAI-live/models/piper/en_US-lessac-medium.onnx")
KOKORO_MODEL = os.getenv("KOKORO_MODEL", "UnivAI-live/models/kokoro/kokoro-v1.0.onnx")
KOKORO_VOICES = os.getenv("KOKORO_VOICES", "UnivAI-live/models/kokoro/voices-v1.0.bin")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")


def _resolve(relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else ROOT / path


class TTSEngine:
    """render() returns one sentence of float32 PCM at self.sample_rate."""

    name = "none"
    sample_rate = 24000

    def render(self, text: str) -> np.ndarray:
        raise NotImplementedError


class KokoroTTS(TTSEngine):
    name = "kokoro"
    sample_rate = 24000

    def __init__(self) -> None:
        from kokoro_onnx import Kokoro

        model, voices = _resolve(KOKORO_MODEL), _resolve(KOKORO_VOICES)
        if not model.exists() or not voices.exists():
            raise RuntimeError(
                f"Kokoro files missing ({model}, {voices}). Download them from the "
                "kokoro-onnx releases page, or set TTS_ENGINE=piper."
            )
        self.kokoro = Kokoro(str(model), str(voices))
        self.voice = KOKORO_VOICE
        print(f"[tts] Kokoro '{self.voice}' ready ({describe()})")

    def render(self, text: str) -> np.ndarray:
        samples, rate = self.kokoro.create(text, voice=self.voice, speed=1.0, lang="en-us")
        self.sample_rate = rate
        return np.asarray(samples, dtype=np.float32)


class PiperTTS(TTSEngine):
    name = "piper"

    def __init__(self) -> None:
        from piper import PiperVoice

        model = _resolve(PIPER_MODEL)
        if not model.exists():
            raise RuntimeError(
                f"Piper voice not found at {model}. Download one from "
                "huggingface.co/rhasspy/piper-voices and set PIPER_MODEL."
            )
        self.voice = PiperVoice.load(str(model), use_cuda=_use_piper_cuda())
        self.sample_rate = self.voice.config.sample_rate
        print(f"[tts] Piper '{model.name}' ready ({self.sample_rate} Hz)")

    def render(self, text: str) -> np.ndarray:
        chunks = [c.audio_int16_array for c in self.voice.synthesize(text)]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32) / 32768.0


class CoquiTTS(TTSEngine):
    name = "coqui-xtts-v2"
    sample_rate = 24000

    def __init__(self) -> None:
        from TTS.api import TTS as CoquiAPI

        self.model = CoquiAPI(
            "tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False
        ).to(device())
        print(f"[tts] XTTS-v2 loaded on {describe()}")
        self.speaker = os.getenv("TTS_SPEAKER", "Ana Florence")
        self.language = os.getenv("TTS_LANGUAGE", "en")

    def render(self, text: str) -> np.ndarray:
        wav = self.model.tts(text=text, speaker=self.speaker, language=self.language)
        return np.asarray(wav, dtype=np.float32)


ENGINES = {"kokoro": KokoroTTS, "piper": PiperTTS, "coqui": CoquiTTS}


def _use_piper_cuda() -> bool:
    """Use ONNX CUDA only when explicitly requested and loadable."""
    preference = os.getenv("PIPER_DEVICE", os.getenv("DEVICE", "auto")).strip().lower()
    if preference not in {"auto", "cpu", "cuda"}:
        raise ValueError("PIPER_DEVICE must be one of: auto, cpu, cuda")
    if preference != "cuda":
        return False
    if not cuda_available():
        print("[tts] Piper CUDA requested but CUDA is unavailable to PyTorch; using CPU")
        return False
    return _onnx_cuda_provider_loadable()


def _onnx_cuda_provider_loadable() -> bool:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        print(f"[tts] Piper CUDA requested but ONNX Runtime is unavailable ({exc}); using CPU")
        return False
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("[tts] Piper CUDA requested but ONNX Runtime has no CUDA provider; using CPU")
        return False

    capi = Path(ort.__file__).resolve().parent / "capi"
    library_name = (
        "onnxruntime_providers_cuda.dll"
        if sys.platform == "win32"
        else "libonnxruntime_providers_cuda.so"
    )
    libraries = list(capi.glob(library_name))
    if not libraries:
        print("[tts] Piper CUDA requested but its ONNX provider library is missing; using CPU")
        return False
    try:
        if sys.platform == "win32":
            with os.add_dll_directory(str(capi)):
                ctypes.WinDLL(str(libraries[0]))
        else:
            ctypes.CDLL(str(libraries[0]))
    except OSError as exc:
        print(f"[tts] Piper CUDA provider dependencies are unavailable ({exc}); using CPU")
        return False
    return True


def _measure(engine: TTSEngine) -> float:
    """Render one test sentence and report the real speed. Returns elapsed seconds."""
    started = time.perf_counter()
    audio = engine.render("This is a check of the lecturer's voice.")
    elapsed = time.perf_counter() - started
    speech = len(audio) / engine.sample_rate
    ratio = speech / elapsed if elapsed else float("inf")
    print(
        f"[tts] {engine.name}: {speech:.1f}s of speech in {elapsed:.2f}s "
        f"({ratio:.1f}x realtime, gate {LATENCY_GATE_S}s)"
    )
    return elapsed


def load_live_engine() -> TTSEngine:
    """Load the low-latency engine used for narration and live answers."""
    if TTS_LIVE_ENGINE and TTS_LIVE_ENGINE in ENGINES and TTS_LIVE_ENGINE != TTS_ENGINE:
        engine: TTSEngine = ENGINES[TTS_LIVE_ENGINE]()
        _measure(engine)
        return engine
    return load_engine()


def load_engine() -> TTSEngine:
    """Load the configured engine and hold it to the latency gate."""
    factory = ENGINES.get(TTS_ENGINE, KokoroTTS)
    engine: TTSEngine = factory()

    elapsed = _measure(engine)

    if elapsed > LATENCY_GATE_S and engine.name != "piper":
        print(
            f"[tts] GATE FAILED - {engine.name} is too slow to lecture live on this "
            f"machine. Falling back to Piper. Set TTS_ENGINE=piper to make it permanent."
        )
        try:
            return PiperTTS()
        except Exception as exc:
            print(f"[tts] Piper unavailable ({exc}); staying on {engine.name}.")

    return engine
