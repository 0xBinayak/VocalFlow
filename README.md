<p align="center">
  <h1 align="center">VocalFlow</h1>
  <p align="center">Local voice cloning, custom voices, voice design, and audio transcription — all in one web app.</p>
</p>

<p align="center">
  <a href="https://pypi.org/project/vocalflow/"><img src="https://img.shields.io/pypi/v/vocalflow?color=818cf8&style=flat-square" alt="PyPI"></a>
  <a href="https://github.com/0xBinayak/VocalFlow/actions"><img src="https://img.shields.io/github/actions/workflow/status/0xBinayak/VocalFlow/ci.yml?style=flat-square&label=CI" alt="CI"></a>
  <a href="https://github.com/0xBinayak/VocalFlow/blob/main/LICENSE"><img src="https://img.shields.io/github/license/0xBinayak/VocalFlow?style=flat-square" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.11-blue?style=flat-square" alt="Python">
</p>

---

## What is VocalFlow?

VocalFlow is a self-hosted web application that runs entirely on your local machine. It provides four core capabilities powered by the full Qwen3-TTS model family:

- **Voice Cloning** — upload a short reference audio clip and clone the speaker's voice for new text
- **Custom Voice** — use 9 premium preset speakers with instruction control for emotion, tone, and style
- **Voice Design** — create entirely new voices from natural language descriptions, no reference audio needed
- **Audio Transcription** — transcribe audio files with word-level timestamps

No cloud APIs. No data leaves your machine. Just your GPU doing the work.

## Features

### Voice Cloning (Base model)

- Upload a reference audio clip (3-10 seconds) and clone the speaker's voice
- Optionally provide a transcript of the reference audio for higher quality (ICL mode)
- Choose between 1.7B (best quality) and 0.6B (faster) model sizes
- Save cloned voice prompts as `.pt` files and reuse them without re-uploading audio
- 10+ languages: English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- Auto language detection

### Custom Voice (9 preset speakers)

- **Speakers:** Vivian, Serena, Uncle Fu, Dylan, Eric (Chinese), Ryan, Aiden (English), Ono Anna (Japanese), Sohee (Korean)
- Instruction control (1.7B only): tell the model *how* to speak — "say it angrily", "whisper softly", "speak with excitement"
- Choose between 1.7B (instructions + speakers) and 0.6B (speakers only) model sizes

### Voice Design (create a voice from text)

- Describe the voice you want in natural language: age, gender, tone, emotion, speaking style
- No reference audio needed — the model invents a completely new voice
- Uses the 1.7B VoiceDesign model

### Audio Transcription

- Upload any audio file (`.wav`, `.mp3`, `.flac`, `.ogg`)
- Choose from 6 Whisper model sizes (tiny to large) based on your speed/accuracy needs
- Full transcript with word-level timestamps
- Download as structured JSON
- Auto-detect language or specify manually

### Smart Model Management

- **Automatic model switching** — only one model is kept in VRAM at a time; switching between models (e.g. Voice Clone → Transcribe) automatically unloads the previous model and frees GPU memory
- **Manual unload** — one-click button to release all GPU memory

### Interface

- Clean dark-themed UI built with [Gradio](https://www.gradio.app/)
- Compact sidebar with live-updating generation history
- Per-item download and delete buttons in the history sidebar
- Color-coded history entries per mode (Clone, Custom Voice, Voice Design, Transcribe)
- Inline audio playback
- Flash Attention 2 support for faster inference

## Models

### Qwen3-TTS (Speech Synthesis)

| Model | Parameters | Purpose |
|-------|-----------|---------|
| [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | 1.7B | Voice cloning from reference audio |
| [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | 0.6B | Faster voice cloning |
| [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | 1.7B | 9 preset speakers + instruction control |
| [Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | 0.6B | 9 preset speakers (no instructions) |
| [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | 1.7B | Create voices from text descriptions |

- All models run in `bfloat16` with SDPA attention
- Flash Attention 2 enabled when available
- Downloaded automatically on first use
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

- **Python 3.11** and [FFmpeg](https://ffmpeg.org/) (`winget install ffmpeg` / `brew install ffmpeg` / `apt install ffmpeg`)
- [SoX](https://sox.sourceforge.net/) (`winget install sox` or install from sourceforge)
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

> **First run:** The TTS and Whisper models will be downloaded automatically on first use. Only the model you select is downloaded — you don't need all of them.

## Project Structure

```
VocalFlow/
├── app.py              # Main web application (Gradio UI + all TTS modes)
├── transcribe.py       # Whisper transcription module (also usable as CLI)
├── main.py             # CLI entry points (vocalflow, vocalflow-transcribe)
├── pyproject.toml      # Dependencies and build config
├── uv.lock             # Locked dependency versions
├── models/             # Cached model weights (gitignored, created at runtime)
├── audio/              # Generated speech files (gitignored, created at runtime)
├── voices/             # Saved voice prompt files (gitignored, created at runtime)
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

**Areas where help is appreciated:** streaming TTS, batch processing, audio post-processing, UI improvements, and testing.

Found a bug? [Open an issue](https://github.com/0xBinayak/VocalFlow/issues).

## License

MIT

## Acknowledgments

- [Qwen3-TTS](https://huggingface.co/Qwen) by Alibaba Cloud — voice cloning, custom voice, and voice design models
- [OpenAI Whisper](https://github.com/openai/whisper) — speech recognition
- [Gradio](https://www.gradio.app/) — web UI framework
