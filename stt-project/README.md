# STT test project

This folder contains a local microphone-to-transcript demo for testing Faster Whisper with Silero VAD on Ubuntu.

## Requirements

- Ubuntu 22.04
- Python 3.11
- NVIDIA GPU with CUDA
- ffmpeg

## Setup

```bash
cd /home/mo/xtts-stream/stt-project
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1 faster-whisper==1.1.1 ctranslate2==4.5.0 sounddevice numpy scipy soundfile webrtcvad silero-vad librosa pyyaml tqdm
```

## Run file transcription

```bash
python transcribe_file.py
```

## Run live microphone transcription

```bash
python live_transcribe.py
```

## Notes

- The model will be cached under the models folder.
- The live transcription loop is intentionally simple and can be connected later to XTTS or a WebSocket API.
