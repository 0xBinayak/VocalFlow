# VocalFlow

Local text-to-speech and audio transcription web app.

## Setup

```bash
uv sync
uv run app.py
```

Open http://localhost:5001 in your browser.

## Features

### Speech (TTS)
- **Voice Design** — describe the voice you want (age, accent, tone)
- **Voice Cloning** — upload a reference audio clip to clone a speaker
- Multi-language support

### Transcribe
- Upload any audio file and get word-level timestamps
- Powered by OpenAI Whisper (tiny → large models)
- Copy transcript or download as JSON
- Auto-detect language or specify manually

### General
- Sidebar with live-updating history
- Playback and download from history
- GPU-accelerated (CUDA)
