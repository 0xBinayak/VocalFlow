import json
import os
import sys
import time
from pathlib import Path

import torch
import whisper

_MODEL_DIR = str(Path(__file__).resolve().parent / "models" / "whisper")


def get_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU detected: {name} ({mem:.1f} GB)")
        return "cuda"
    print("No CUDA GPU found — falling back to CPU")
    return "cpu"


def transcribe_words(
    audio_path: str,
    model_size: str = "base",
    device: str | None = None,
    language: str | None = None,
) -> list[dict]:
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    device = device or get_device()

    print(f"Loading Whisper '{model_size}' model on {device.upper()}...")
    model = whisper.load_model(model_size, device=device, download_root=_MODEL_DIR)

    print(f"Transcribing: {audio_path}")
    t0 = time.perf_counter()

    transcribe_opts: dict = {"word_timestamps": True, "verbose": False}
    if language:
        transcribe_opts["language"] = language

    result = model.transcribe(audio_path, **transcribe_opts)
    elapsed = time.perf_counter() - t0

    words_data = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            words_data.append({
                "word": w["word"].strip(),
                "start": round(w["start"], 3),
                "end": round(w["end"], 3),
            })

    out_path = os.path.splitext(audio_path)[0] + "_words.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(words_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(words_data)} words to {out_path} in {elapsed:.1f}s")
    return words_data


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="transcriber",
        description="Transcribe audio to word-level timestamps using Whisper.",
    )
    parser.add_argument("audio", help="Path to the audio file")
    parser.add_argument(
        "-m", "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "-d", "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Force compute device (default: auto-detect)",
    )
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="Language code, e.g. 'en' (default: auto-detect)",
    )
    args = parser.parse_args()
    transcribe_words(args.audio, args.model, args.device, args.language)


if __name__ == "__main__":
    main()
