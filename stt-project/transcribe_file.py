import os
from faster_whisper import WhisperModel

MODEL_SIZE = "large-v3"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

print("Loading Whisper...")
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16", download_root=MODEL_DIR)

input_wav = os.path.join(os.path.dirname(__file__), "sample.wav")
if not os.path.exists(input_wav):
    raise FileNotFoundError(f"Missing sample audio: {input_wav}")

segments, info = model.transcribe(input_wav, beam_size=5, vad_filter=True)
print("Detected language:", info.language)
print("Language probability:", getattr(info, "language_probability", None))
print("Transcription:")
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
