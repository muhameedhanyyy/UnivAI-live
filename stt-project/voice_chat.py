import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from silero_vad import get_speech_timestamps, load_silero_vad

os.environ.setdefault("COQUI_TOS_AGREED", "1")

for candidate in [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "TTS")),
]:
    if os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, "TTS")):
        sys.path.insert(0, candidate)
        break

from TTS.api import TTS

MODEL_SIZE = "tiny.en"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
SAMPLE_RATE = 16000
RECORD_SECONDS = 3.0
os.makedirs(MODEL_DIR, exist_ok=True)

speaker_wav = "/home/mo/xtts-stream/TTS/tests/inputs/example_1.wav"

print("Loading Whisper...")
whisper_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", download_root=MODEL_DIR)
print("Loading Silero VAD...")
vad_model = load_silero_vad(onnx=False)
print("Loading XTTS...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)


def build_reply(text: str) -> str:
    t = text.lower()
    if any(x in t for x in ["hello", "hi", "hey"]):
        return "Hello! I am your local voice tutor. Ask me about what you need."
    if "c++" in t or "cpp" in t:
        return "C++ is a fast, compiled language used for systems programming, game engines, and performance-critical software. It supports object-oriented and low-level programming features."
    if "python" in t:
        return "Python is a high-level programming language used for automation, web apps, and AI. Would you like a short example?"
    if any(x in t for x in ["bye", "exit", "stop", "shutdown"]):
        return "Goodbye. I will stop listening now."
    return f"I heard: {text}. I can help with C++ or Python and general programming questions."


def play_audio(wav_path: str) -> None:
    subprocess.run(["aplay", "-q", wav_path], check=False)


def record_audio(seconds: float) -> np.ndarray:
    recording = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    return recording[:, 0].astype(np.float32) / 32768.0


def transcribe_audio(audio: np.ndarray) -> str:
    speech_timestamps = get_speech_timestamps(audio, vad_model, sampling_rate=SAMPLE_RATE)
    if not speech_timestamps:
        return ""

    segments, _ = whisper_model.transcribe(audio, beam_size=5, vad_filter=False)
    text = " ".join(segment.text for segment in segments).strip()
    return text


def run_once(text: str | None = None) -> bool:
    if text is None:
        print("Listening...")
        audio = record_audio(RECORD_SECONDS)
        text = transcribe_audio(audio)
        print(f"Detected: {text or '<no speech>'}")
    else:
        print(f"Detected: {text}")

    if not text:
        return True

    if any(x in text.lower() for x in ["bye", "exit", "stop", "shutdown"]):
        reply = "Goodbye. I will stop listening now."
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        tts.tts_to_file(text=reply, speaker_wav=speaker_wav, language="en", file_path=out_path)
        play_audio(out_path)
        os.remove(out_path)
        return False

    reply = build_reply(text)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name
    tts.tts_to_file(text=reply, speaker_wav=speaker_wav, language="en", file_path=out_path)
    play_audio(out_path)
    os.remove(out_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple local voice conversation demo")
    parser.add_argument("--text", help="Optional text input instead of microphone recording")
    args = parser.parse_args()

    while True:
        keep_going = run_once(args.text)
        if args.text is not None:
            break
        if not keep_going:
            break
        time.sleep(0.2)


if __name__ == "__main__":
    main()
