<p align="center">
  <h1 align="center">VocalFlow</h1>
  <p align="center">Local TTS, voice cloning, and transcription for Windows.</p>
</p>

<p align="center">
  <a href="https://pypi.org/project/vocalflow/"><img src="https://img.shields.io/pypi/v/vocalflow?color=818cf8&style=flat-square" alt="PyPI"></a>
  <a href="https://github.com/0xBinayak/VocalFlow/actions"><img src="https://img.shields.io/github/actions/workflow/status/0xBinayak/VocalFlow/ci.yml?style=flat-square&label=CI" alt="CI"></a>
  <a href="https://github.com/0xBinayak/VocalFlow/blob/main/LICENSE"><img src="https://img.shields.io/github/license/0xBinayak/VocalFlow?style=flat-square" alt="License"></a>
  <img src="https://img.shields.io/badge/platform-Windows-0078D6?style=flat-square" alt="Windows">
  <img src="https://img.shields.io/badge/python-3.11-blue?style=flat-square" alt="Python">
</p>

---

Self-hosted web app that runs entirely on your machine. No cloud APIs, no data leaves your PC.

## Features

- **Voice Cloning** — clone any voice from a short audio clip (3-10s), save & reuse voice prompts
- **Custom Voice** — 9 preset speakers with emotion/tone control ("say it angrily", "whisper softly")
- **Voice Design** — create new voices from text descriptions, no reference audio needed
- **Transcription** — Whisper-powered transcription with word-level timestamps, 6 model sizes
- **Smart GPU** — automatic model load/unload between switches, only one model in VRAM at a time
- **Flash Attention 2** for faster inference
- 10+ languages with auto-detection

## Requirements

- **Windows 10/11** with a CUDA GPU (6+ GB VRAM)
- **Python 3.11**
- [FFmpeg](https://ffmpeg.org/) — `winget install ffmpeg`
- [SoX](https://sox.sourceforge.net/) — `winget install sox`

## Quick Start

```
pip install vocalflow
vocalflow
```

Open **http://localhost:5001**. Models download automatically on first use.

### From source

```
git clone https://github.com/0xBinayak/VocalFlow.git
cd VocalFlow
uv sync
uv run app.py
```

For auto-reload during development: `uv run gradio app.py`

## Models

| Model | Params | Use |
|-------|--------|-----|
| Qwen3-TTS-1.7B-Base | 1.7B | Voice cloning (best quality) |
| Qwen3-TTS-0.6B-Base | 0.6B | Voice cloning (faster) |
| Qwen3-TTS-1.7B-CustomVoice | 1.7B | Preset speakers + instructions |
| Qwen3-TTS-0.6B-CustomVoice | 0.6B | Preset speakers only |
| Qwen3-TTS-1.7B-VoiceDesign | 1.7B | Voice from text description |
| Whisper (tiny-turbo) | 39M-1.5B | Transcription |

All TTS models run in bfloat16 with SDPA/Flash Attention. Whisper falls back to CPU if no GPU.

## Contributing

1. Fork & clone, `uv sync`, create a branch
2. Ensure `uvx ruff check app.py transcribe.py main.py` passes
3. Open a PR against `main`

[Open an issue](https://github.com/0xBinayak/VocalFlow/issues) if you find a bug.

## License

MIT

## Acknowledgments

[Qwen3-TTS](https://huggingface.co/Qwen) | [OpenAI Whisper](https://github.com/openai/whisper) | [Gradio](https://www.gradio.app/)
