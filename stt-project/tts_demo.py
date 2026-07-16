import os

os.environ.setdefault("COQUI_TOS_AGREED", "1")

from TTS.api import TTS

speaker_wav = "/home/mo/xtts-stream/TTS/tests/inputs/example_1.wav"
out_wav = os.path.join(os.path.dirname(__file__), "output.wav")

print("Loading XTTS v2...")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)

print("Generating speech...")
tts.tts_to_file(
    text="This is a simple XTTS demo from the research branch.",
    speaker_wav=speaker_wav,
    language="en",
    file_path=out_wav,
)

print(f"Saved audio to {out_wav}")
