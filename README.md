<p align="center">
  <h1 align="center">VocalFlow</h1>
  <p align="center">Local voice cloning and audio transcription — all in one web app.</p>
</p>

<p align="center">
  <a href="https://pypi.org/project/vocalflow/"><img src="https://img.shields.io/pypi/v/vocalflow?color=818cf8&style=flat-square" alt="PyPI"></a>
  <a href="https://github.com/0xBinayak/VocalFlow/actions"><img src="https://img.shields.io/github/actions/workflow/status/0xBinayak/VocalFlow/ci.yml?style=flat-square&label=CI" alt="CI"></a>
  <a href="https://github.com/0xBinayak/VocalFlow/blob/main/LICENSE"><img src="https://img.shields.io/github/license/0xBinayak/VocalFlow?style=flat-square" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square" alt="Python">
</p>

---

## What is VocalFlow?

VocalFlow is a self-hosted web application that runs entirely on your local machine. It provides two core capabilities:

- **Voice Cloning (TTS)** — upload a short reference audio clip and clone the speaker's voice for new text
- **Audio Transcription** — transcribe audio files with word-level timestamps

No cloud APIs. No data leaves your machine. Just your GPU doing the work.

## Features

### Voice Cloning

- Upload a reference audio clip (3-10 seconds) and clone the speaker's voice
- Optionally provide a transcript of the reference audio for higher quality
- 10+ languages: English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- Auto language detection
- Download generated audio as `.wav`

### Audio Transcription

- Upload any audio file (`.wav`, `.mp3`, `.flac`, `.ogg`)
- Choose from 6 Whisper model sizes (tiny to large) based on your speed/accuracy needs
- Full transcript with word-level timestamps
- Copy transcript to clipboard or download as structured JSON
- Auto-detect language or specify manually

### Interface

- Clean dark-themed UI built with [FastHTML](https://github.com/AnswerDotAI/fasthtml) + HTMX
- Sidebar with live-updating generation history
- Inline audio playback from history
- One-click downloads with human-readable filenames

## Models

### Qwen3-TTS (Voice Cloning)

| Model | Parameters | Purpose |
|-------|-----------|---------|
| [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | 1.7B | Voice cloning from reference audio |

- Runs in `bfloat16` with SDPA attention for efficient inference
- Downloaded automatically on first use (~3.5 GB)
- Requires a CUDA GPU with at least 6 GB VRAM

### OpenAI Whisper (Transcription)

| Model | Parameters | VRAM | Relative Speed |
|-------|-----------|------|----------------|
| `tiny` | 39M | ~1 GB | ~32x realtime |
| `base` | 74M | ~1 GB | ~16x realtime |
| `small` | 244M | ~2 GB | ~6x realtime |
| `medium` | 769M | ~5 GB | ~2x realtime |
| `large` | 1.5B | ~10 GB | ~1x realtime |
| `turbo` | 1.5B | ~6 GB | ~8x realtime |

- Models are downloaded automatically on first use
- Falls back to CPU if no CUDA GPU is detected

## Requirements

- Python 3.11+ and [FFmpeg](https://ffmpeg.org/) (`winget install ffmpeg` / `brew install ffmpeg` / `apt install ffmpeg`)
- CUDA GPU with 6+ GB VRAM for TTS (transcription can run on CPU)

## Quick Start

### Install

```bash
pip install vocalflow
```

### Launch the web app

```bash
vocalflow
```

Then open **http://localhost:5001** in your browser.

### From source

```bash
git clone https://github.com/0xBinayak/VocalFlow.git
cd VocalFlow
uv sync
uv run app.py
```

> **First run:** The TTS and Whisper models will be downloaded automatically. This may take a few minutes depending on your connection.

## Project Structure

```
VocalFlow/
├── app.py              # Main web application (FastHTML + routes + UI)
├── transcribe.py       # Whisper transcription module (also usable as CLI)
├── main.py             # CLI entry points (vocalflow, vocalflow-transcribe)
├── pyproject.toml      # Dependencies and build config
├── uv.lock             # Locked dependency versions
├── models/             # Cached model weights (gitignored, created at runtime)
├── audio/              # Generated speech files (gitignored, created at runtime)
├── uploads/            # Temporary uploaded files (gitignored, created at runtime)
├── transcripts/        # Transcription JSON files (gitignored, created at runtime)
└── .github/workflows/
    ├── ci.yml          # Lint + syntax checks on push/PR
    ├── release.yml     # Auto GitHub Release on version tags
    └── publish.yml     # Auto PyPI publish on version tags
```

## Contributing

Contributions are welcome!

1. Fork the repo and clone it
2. `uv sync` to install dependencies
3. Create a branch, make changes, ensure `uvx ruff check app.py transcribe.py main.py` passes
4. Open a Pull Request against `main`

**Areas where help is appreciated:** additional TTS backends, speaker diarization, batch processing, audio post-processing, UI improvements, and testing.

Found a bug? [Open an issue](https://github.com/0xBinayak/VocalFlow/issues).

## License

MIT

## Acknowledgments

- [Qwen3-TTS](https://huggingface.co/Qwen) by Alibaba Cloud — voice cloning model
- [OpenAI Whisper](https://github.com/openai/whisper) — speech recognition
- [FastHTML](https://github.com/AnswerDotAI/fasthtml) — Python web framework
- [HTMX](https://htmx.org/) — frontend interactivity
