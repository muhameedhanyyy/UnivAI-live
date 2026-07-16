# STT and XTTS test project

This folder contains a local prototype for testing:

- Speech-to-text with Faster Whisper + Silero VAD
- Text-to-speech with XTTS v2

## Requirements

- Ubuntu 22.04
- Python 3.11
- NVIDIA GPU with CUDA
- ffmpeg

## Setup

```bash
cd /home/mo/xtts-stream/UnivAI-live/stt-project
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Run file transcription

```bash
python transcribe_file.py
```

## Run live microphone transcription

```bash
python live_transcribe.py
```

## Run XTTS demo

```bash
python tts_demo.py
```

## Run the voice-chat prototype

```bash
python voice_chat.py
```

## Notes

- The Whisper model cache will be stored under the models folder.
- The voice-chat loop is intentionally simple and can be connected later to a larger tutor flow.
