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

Self-hosted web app powered by [Qwen3-TTS](https://huggingface.co/Qwen) and [Whisper](https://github.com/openai/whisper). Everything runs locally — no cloud APIs, no data leaves your machine.

## Features

- **Voice Cloning** — clone any voice from a short audio clip, save and reuse voice prompts
- **Custom Voice** — 9 preset speakers with emotion and tone control
- **Voice Design** — create new voices from natural language descriptions
- **Transcription** — word-level timestamps, 6 model sizes, auto language detection
- **Smart GPU** — automatic model switching with VRAM cleanup, Flash Attention 2
- **10+ languages** including English, Chinese, Japanese, Korean, and more

## Requirements

- **Windows 10/11** with a CUDA GPU (6 GB+ VRAM)
- **Python 3.11** &ensp;|&ensp; [FFmpeg](https://ffmpeg.org/) `winget install ffmpeg` &ensp;|&ensp; [SoX](https://sox.sourceforge.net/) `winget install sox`

## Quick Start

```
pip install vocalflow
vocalflow
```

Open **http://localhost:5001** &mdash; models download automatically on first use.

### From source

```
git clone https://github.com/0xBinayak/VocalFlow.git && cd VocalFlow
uv sync && uv run app.py
```

Dev mode with auto-reload: `uv run gradio app.py`

## Contributing

Fork, `uv sync`, branch, ensure `uvx ruff check app.py transcribe.py main.py` passes, PR.
&ensp;[Report a bug](https://github.com/0xBinayak/VocalFlow/issues)

## License

MIT &ensp;|&ensp; [Qwen3-TTS](https://huggingface.co/Qwen) &ensp;|&ensp; [OpenAI Whisper](https://github.com/openai/whisper) &ensp;|&ensp; [Gradio](https://www.gradio.app/)
