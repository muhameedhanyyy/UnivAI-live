import os
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, get_speech_timestamps

MODEL_SIZE = "large-v3"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
SAMPLE_RATE = 16000
CHUNK_SECONDS = 1.0

print("Loading Whisper...")
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16", download_root=MODEL_DIR)
print("Loading Silero VAD...")
vad_model = load_silero_vad(onnx=False)

q = queue.Queue()
stop_event = threading.Event()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


def capture_loop():
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=audio_callback):
        while not stop_event.is_set():
            time.sleep(0.1)


def transcribe_loop():
    while not stop_event.is_set():
        try:
            chunk = q.get(timeout=0.2)
        except queue.Empty:
            continue

        audio = chunk[:, 0].astype(np.float32) / 32768.0
        speech_timestamps = get_speech_timestamps(audio, vad_model, sampling_rate=SAMPLE_RATE)
        if not speech_timestamps:
            continue

        print("\nListening...")
        print("Speech detected...")
        print("Transcribing...")
        segments, _ = model.transcribe(audio, vad_filter=False, word_timestamps=False)
        text = " ".join(segment.text for segment in segments).strip()
        if text:
            print(f"Transcript: {text}")
        else:
            print("Transcript: <no speech>")


threading.Thread(target=capture_loop, daemon=True).start()
threading.Thread(target=transcribe_loop, daemon=True).start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    stop_event.set()
    print("\nStopped.")
