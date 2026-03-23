import gc
import os
import re
import uuid
import json
import time
from dataclasses import asdict
from pathlib import Path
from datetime import datetime

# The `sox` Python package (pulled in by qwen-tts) checks for the SoX binary
# at import time via os.popen, which on Windows goes through cmd.exe and fails
# when cmd AutoRun is set (DOSKEY etc.).  We detect SoX reliably with
# shutil.which, then import sox with stderr silenced and fix the flag.
import shutil
if shutil.which("sox"):
    import contextlib
    import subprocess
    with open(os.devnull, "w") as _devnull, contextlib.redirect_stderr(_devnull):
        # Monkey-patch os.popen for the duration of the sox import so the
        # broken cmd.exe check doesn't spam the console.
        _orig_popen = os.popen
        os.popen = lambda *a, **k: subprocess.Popen(
            ["sox", "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).stdout
        import sox as _sox_mod
        os.popen = _orig_popen
    _sox_mod.NO_SOX = False

import torch
import gradio as gr
import soundfile as sf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "audio"
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
VOICES_DIR = BASE_DIR / "voices"
HISTORY_FILE = BASE_DIR / "history.json"
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
TRANSCRIPT_DIR.mkdir(exist_ok=True)
VOICES_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def _local_snapshot(repo_id: str) -> str:
    """Resolve a HF repo id to its local snapshot path, or return repo_id as fallback."""
    d = MODEL_DIR / ("models--" + repo_id.replace("/", "--"))
    refs = d / "refs" / "main"
    if refs.exists():
        snap = d / "snapshots" / refs.read_text(encoding="utf-8").strip()
        if snap.exists():
            return str(snap.resolve())
    return repo_id


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------
def _load_history() -> list:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_history(entries: list):
    HISTORY_FILE.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def _add_history(filename: str, text: str, mode: str, language: str, voice_desc: str = ""):
    entries = _load_history()
    entries.insert(0, {
        "file": filename,
        "text": text[:120],
        "mode": mode,
        "language": language,
        "voice": voice_desc[:80],
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    _save_history(entries[:50])


def _slug(text: str, max_len: int = 50) -> str:
    s = text.strip()[:max_len].lower()
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s_]+', '-', s).strip('-')
    return s or "untitled"


# ---------------------------------------------------------------------------
# TTS Model loading (lazy, per model type + size)
# Automatic unload: only one model (TTS or Whisper) is kept in memory at a
# time.  Switching between models triggers an automatic unload of the
# previous one so VRAM is reclaimed without manual intervention.
# ---------------------------------------------------------------------------
_models = {}
_active_tts_key = None          # track which TTS model is currently loaded

MODEL_REPO_MAP = {
    ("base", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ("base", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ("custom_voice", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    ("custom_voice", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    ("voice_design", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}


def _release_gpu():
    """Run garbage collection and release cached CUDA memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _unload_tts(reason: str = ""):
    """Unload the currently loaded TTS model (if any) and free VRAM."""
    global _active_tts_key
    if _active_tts_key is not None and _active_tts_key in _models:
        label = f"{_active_tts_key[0]} {_active_tts_key[1]}"
        del _models[_active_tts_key]
        _active_tts_key = None
        _release_gpu()
        print(f"[model] Auto-unloaded TTS ({label}){f' — {reason}' if reason else ''}")


def _unload_whisper(reason: str = ""):
    """Unload the currently loaded Whisper model (if any) and free VRAM."""
    global _whisper_model, _whisper_size
    if _whisper_model is not None:
        old_size = _whisper_size
        _whisper_model = None
        _whisper_size = None
        _release_gpu()
        print(f"[model] Auto-unloaded Whisper ({old_size}){f' — {reason}' if reason else ''}")


def get_tts_model(model_type: str, model_size: str):
    global _active_tts_key
    key = (model_type, model_size)

    # Already loaded — nothing to do
    if key == _active_tts_key and key in _models:
        return _models[key]

    repo_id = MODEL_REPO_MAP.get(key)
    if repo_id is None:
        raise gr.Error(f"No model available for {model_type} / {model_size}")

    # --- automatic unload before loading the new model ---
    _unload_whisper(reason=f"switching to TTS {model_type} {model_size}")
    _unload_tts(reason=f"switching to {model_type} {model_size}")

    from qwen_tts import Qwen3TTSModel
    path = _local_snapshot(repo_id)
    print(f"[model] Loading {model_type} {model_size} from {path}")
    _models[key] = Qwen3TTSModel.from_pretrained(
        path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    _active_tts_key = key
    print(f"[model] {model_type} {model_size} ready.")
    return _models[key]


# ---------------------------------------------------------------------------
# Whisper model loading (lazy)
# ---------------------------------------------------------------------------
_whisper_model = None
_whisper_size = None


def get_whisper_model(size: str = "base"):
    global _whisper_model, _whisper_size

    # Already loaded at the requested size — nothing to do
    if _whisper_model is not None and _whisper_size == size:
        return _whisper_model

    # --- automatic unload before loading the new model ---
    _unload_tts(reason=f"switching to Whisper {size}")
    _unload_whisper(reason=f"switching to Whisper {size}")

    import whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] Loading Whisper '{size}' on {device.upper()} ...")
    _whisper_model = whisper.load_model(size, device=device, download_root=str(MODEL_DIR / "whisper"))
    _whisper_size = size
    print("[model] Whisper ready.")
    return _whisper_model


# ---------------------------------------------------------------------------
# Unload all models from GPU
# ---------------------------------------------------------------------------
def unload_all_models():
    global _active_tts_key
    count = len(_models) + (1 if _whisper_model is not None else 0)
    if count == 0:
        return 0
    _unload_tts()
    _unload_whisper()
    _models.clear()
    _active_tts_key = None
    _release_gpu()
    print(f"[model] Unloaded {count} model(s), GPU memory released.")
    return count


# ---------------------------------------------------------------------------
# Clear history + associated files + unload models
# ---------------------------------------------------------------------------
def delete_history_item(filename: str) -> str:
    """Delete a single history entry and its associated file."""
    if not filename:
        return render_history_html()
    entries = _load_history()
    new_entries = [e for e in entries if e.get("file") != filename]
    if len(new_entries) < len(entries):
        for d in (OUTPUT_DIR, TRANSCRIPT_DIR):
            p = d / filename
            if p.exists():
                p.unlink()
        _save_history(new_entries)
        print(f"[history] Deleted item: {filename}")
    return render_history_html()


def clear_history_and_files():
    entries = _load_history()
    deleted_files = 0
    for e in entries:
        fname = e.get("file", "")
        if not fname:
            continue
        for d in (OUTPUT_DIR, TRANSCRIPT_DIR):
            p = d / fname
            if p.exists():
                p.unlink()
                deleted_files += 1
    for f in UPLOAD_DIR.iterdir():
        if f.is_file():
            f.unlink()
            deleted_files += 1
    _save_history([])
    print(f"[clear] Deleted {deleted_files} file(s).")
    return render_history_html()


# ---------------------------------------------------------------------------
# History rendering (HTML for sidebar)
# ---------------------------------------------------------------------------
MODE_DISPLAY = {
    "clone": ("C", "clone", "Clone"),
    "custom_voice": ("S", "custom-voice", "Custom Voice"),
    "voice_design": ("D", "design", "Voice Design"),
    "transcribe": ("T", "transcribe", "Transcribe"),
}


def render_history_html() -> str:
    entries = _load_history()
    if not entries:
        return '<div class="hist-empty">No activity yet.</div>'

    items = []
    for e in entries[:30]:
        mode = e.get("mode", "")
        icon, icon_cls, mode_label = MODE_DISPLAY.get(mode, ("?", "clone", mode))

        sub_parts = [e.get("time", ""), mode_label]
        lang = e.get("language", "")
        if lang:
            sub_parts.append(lang)
        voice = e.get("voice", "")
        if voice:
            sub_parts.append(voice[:30])
        sub = "  \u00b7  ".join(sub_parts)

        text_display = e.get("text", "\u2014")
        text_display = text_display.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        sub = sub.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # Build download link
        fname = e.get("file", "")
        if mode == "transcribe":
            dl_path = "/gradio_api/file=" + (TRANSCRIPT_DIR / fname).as_posix()
        else:
            dl_path = "/gradio_api/file=" + (OUTPUT_DIR / fname).as_posix()
        dl_btn = (
            f'<a class="si-dl" href="{dl_path}" download="{fname}" '
            f'title="Download">\u2913</a>'
        ) if fname else ""
        del_btn = (
            f'<button class="si-del" onclick="deleteHistoryItem(\'{fname}\')" '
            f'title="Delete">\u00d7</button>'
        ) if fname else ""

        items.append(f'''<div class="si">
            <div class="si-icon {icon_cls}">{icon}</div>
            <div class="si-meta">
                <div class="si-text">{text_display}</div>
                <div class="si-sub">{sub}</div>
            </div>
            {dl_btn}
            {del_btn}
        </div>''')

    count = len(entries)
    return f'''<div class="sidebar-hdr">
        <h3>History</h3><span class="sidebar-count">{count}</span>
    </div>{''.join(items)}'''


# ---------------------------------------------------------------------------
# Generate speech: Voice Clone (Base model)
# ---------------------------------------------------------------------------
def generate_speech(text, language, ref_audio, ref_text, model_size):
    if not text or not text.strip():
        raise gr.Error("Please enter some text to speak.")
    if ref_audio is None:
        raise gr.Error("Please upload a reference audio file.")

    out_name = f"{uuid.uuid4().hex}.wav"
    out_path = OUTPUT_DIR / out_name
    voice_desc = f"Cloned · {model_size}"

    try:
        model = get_tts_model("base", model_size)
        wavs, sr = model.generate_voice_clone(
            text=text.strip(),
            language=language,
            ref_audio=str(ref_audio),
            ref_text=ref_text.strip() if ref_text and ref_text.strip() else None,
            x_vector_only_mode=not bool(ref_text and ref_text.strip()),
            max_new_tokens=2048,
        )
        sf.write(str(out_path), wavs[0], sr)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Generation failed: {e}")

    if not out_path.exists():
        raise gr.Error("Output file was not created. Please try again.")

    _add_history(out_name, text.strip(), "clone", language, voice_desc)
    return str(out_path), str(out_path), render_history_html()


# ---------------------------------------------------------------------------
# Save / Load voice clone prompt
# ---------------------------------------------------------------------------
def save_voice_prompt(ref_audio, ref_text, model_size):
    if ref_audio is None:
        raise gr.Error("Please upload a reference audio file.")

    try:
        model = get_tts_model("base", model_size)
        x_vec_only = not bool(ref_text and ref_text.strip())
        items = model.create_voice_clone_prompt(
            ref_audio=str(ref_audio),
            ref_text=ref_text.strip() if ref_text and ref_text.strip() else None,
            x_vector_only_mode=x_vec_only,
        )
        payload = {"items": [asdict(it) for it in items]}
        out_name = f"voice_{uuid.uuid4().hex[:8]}.pt"
        out_path = VOICES_DIR / out_name
        torch.save(payload, str(out_path))
        return str(out_path), f"Voice saved: {out_name}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Save failed: {e}")


def generate_from_voice_prompt(text, language, voice_file, model_size):
    if not text or not text.strip():
        raise gr.Error("Please enter some text to speak.")
    if voice_file is None:
        raise gr.Error("Please upload a saved voice file (.pt).")

    from qwen_tts import VoiceClonePromptItem

    try:
        path = voice_file if isinstance(voice_file, str) else voice_file.name
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "items" not in payload:
            raise gr.Error("Invalid voice file format.")

        items = []
        for d in payload["items"]:
            ref_code = d.get("ref_code")
            if ref_code is not None and not torch.is_tensor(ref_code):
                ref_code = torch.tensor(ref_code)
            ref_spk = d.get("ref_spk_embedding")
            if ref_spk is not None and not torch.is_tensor(ref_spk):
                ref_spk = torch.tensor(ref_spk)
            items.append(VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk,
                x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                icl_mode=bool(d.get("icl_mode", True)),
                ref_text=d.get("ref_text"),
            ))

        model = get_tts_model("base", model_size)
        wavs, sr = model.generate_voice_clone(
            text=text.strip(),
            language=language,
            voice_clone_prompt=items,
            max_new_tokens=2048,
        )

        out_name = f"{uuid.uuid4().hex}.wav"
        out_path = OUTPUT_DIR / out_name
        sf.write(str(out_path), wavs[0], sr)

        _add_history(out_name, text.strip(), "clone", language, f"Saved voice · {model_size}")
        return str(out_path), str(out_path), render_history_html()
    except gr.Error:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Generation failed: {e}")


# ---------------------------------------------------------------------------
# Generate speech: Custom Voice (preset speakers)
# ---------------------------------------------------------------------------
SPEAKERS = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
            "Ryan", "Aiden", "Ono_Anna", "Sohee"]


def generate_custom_voice(text, language, speaker, instruct, model_size):
    if not text or not text.strip():
        raise gr.Error("Please enter some text to speak.")
    if not speaker:
        raise gr.Error("Please select a speaker.")

    out_name = f"{uuid.uuid4().hex}.wav"
    out_path = OUTPUT_DIR / out_name
    voice_desc = f"{speaker} · {model_size}"

    try:
        model = get_tts_model("custom_voice", model_size)
        wavs, sr = model.generate_custom_voice(
            text=text.strip(),
            language=language,
            speaker=speaker,
            instruct=instruct.strip() if instruct and instruct.strip() else None,
            max_new_tokens=2048,
        )
        sf.write(str(out_path), wavs[0], sr)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Generation failed: {e}")

    if not out_path.exists():
        raise gr.Error("Output file was not created.")

    _add_history(out_name, text.strip(), "custom_voice", language, voice_desc)
    return str(out_path), str(out_path), render_history_html()


# ---------------------------------------------------------------------------
# Generate speech: Voice Design (create voice from description)
# ---------------------------------------------------------------------------
def generate_voice_design(text, language, instruct):
    if not text or not text.strip():
        raise gr.Error("Please enter some text to speak.")
    if not instruct or not instruct.strip():
        raise gr.Error("Please describe the voice you want.")

    out_name = f"{uuid.uuid4().hex}.wav"
    out_path = OUTPUT_DIR / out_name
    voice_desc = instruct.strip()[:40]

    try:
        model = get_tts_model("voice_design", "1.7B")
        wavs, sr = model.generate_voice_design(
            text=text.strip(),
            language=language,
            instruct=instruct.strip(),
            max_new_tokens=2048,
        )
        sf.write(str(out_path), wavs[0], sr)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Generation failed: {e}")

    if not out_path.exists():
        raise gr.Error("Output file was not created.")

    _add_history(out_name, text.strip(), "voice_design", language, voice_desc)
    return str(out_path), str(out_path), render_history_html()


# ---------------------------------------------------------------------------
# Transcribe audio
# ---------------------------------------------------------------------------
def transcribe_audio(audio_file, whisper_model_size, whisper_lang):
    if audio_file is None:
        raise gr.Error("Please upload an audio file.")

    try:
        model = get_whisper_model(whisper_model_size)
        t0 = time.perf_counter()

        opts = {"word_timestamps": True, "verbose": False}
        if whisper_lang:
            opts["language"] = whisper_lang

        result = model.transcribe(str(audio_file), **opts)
        elapsed = time.perf_counter() - t0

        full_text = result.get("text", "").strip()
        detected_lang = result.get("language", "unknown")

        words_data = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words_data.append({
                    "word": w["word"].strip(),
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                })

        json_name = f"{uuid.uuid4().hex}.json"
        json_path = TRANSCRIPT_DIR / json_name
        json_path.write_text(json.dumps({
            "text": full_text,
            "language": detected_lang,
            "words": words_data,
            "model": whisper_model_size,
            "elapsed": round(elapsed, 1),
            "source": Path(audio_file).name,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        _add_history(json_name, full_text, "transcribe", detected_lang,
                     f"{whisper_model_size} \u00b7 {elapsed:.1f}s")

        meta = f"{len(words_data)} words \u00b7 {detected_lang} \u00b7 {whisper_model_size} \u00b7 {elapsed:.1f}s"
        return full_text, meta, str(json_path), render_history_html()

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Transcription failed: {e}")


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* History sidebar styles */
.sidebar-hdr {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 10px; padding: 0 4px;
}
.sidebar-hdr h3 {
    font-size: .7rem; font-weight: 600; color: #71717a;
    text-transform: uppercase; letter-spacing: .5px; margin: 0;
}
.sidebar-count { font-size: .68rem; color: #52525b; }

.si {
    display: flex; align-items: flex-start; gap: 8px;
    padding: 7px 8px;
    background: #18181b; border: 1px solid transparent; border-radius: 8px;
    margin-bottom: 3px; transition: all .15s; cursor: default;
}
.si:hover { border-color: #27272a; background: #1c1c1f; }

.si-icon {
    width: 26px; height: 26px; border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: .65rem; font-weight: 700; flex-shrink: 0; margin-top: 1px;
}
.si-icon.clone { background: #172554; color: #60a5fa; }
.si-icon.custom-voice { background: #3b0764; color: #c084fc; }
.si-icon.design { background: #713f12; color: #fbbf24; }
.si-icon.transcribe { background: #052e16; color: #4ade80; }

.si-meta { flex: 1; min-width: 0; }
.si-text {
    font-size: .75rem; color: #d4d4d8;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.si-sub { font-size: .62rem; color: #52525b; margin-top: 1px; }

.si-dl {
    width: 22px; height: 22px; border-radius: 5px;
    display: flex; align-items: center; justify-content: center;
    font-size: .82rem; line-height: 1; color: #71717a;
    background: transparent; border: 1px solid transparent;
    text-decoration: none; flex-shrink: 0; margin-top: 2px;
    transition: all .15s;
}
.si-dl:hover { color: #e4e4e7; background: #27272a; border-color: #3f3f46; }

.si-del {
    width: 22px; height: 22px; border-radius: 5px;
    display: flex; align-items: center; justify-content: center;
    font-size: .85rem; line-height: 1; color: #52525b;
    background: transparent; border: 1px solid transparent;
    cursor: pointer; flex-shrink: 0; margin-top: 2px;
    transition: all .15s; padding: 0;
}
.si-del:hover { color: #f87171; background: #2a1215; border-color: #7f1d1d; }

.hist-empty {
    text-align: center; color: #52525b; font-size: .78rem; padding: 30px 16px;
}

/* Brand header */
.brand-header {
    text-align: center; padding: 16px 12px; margin-bottom: 8px;
}
.brand-header h1 {
    font-size: 1.3rem; font-weight: 700; color: #f4f4f5;
    letter-spacing: -.02em; margin: 0;
}
.brand-header p {
    color: #52525b; font-size: .72rem; margin-top: 2px;
}

/* Compact history panel */
.history-panel {
    max-height: calc(100vh - 100px);
    overflow-y: auto;
    padding: 8px;
}
.history-panel::-webkit-scrollbar { width: 4px; }
.history-panel::-webkit-scrollbar-thumb { background: #27272a; border-radius: 4px; }

/* Remove extra padding from the sidebar column */
#sidebar-col { padding: 0 !important; }

/* Transcript meta styling */
.transcript-meta {
    font-size: .82rem; color: #a1a1aa; padding: 6px 0;
}

/* Sidebar action buttons */
.sidebar-actions { padding: 4px 8px !important; gap: 6px !important; }
.clear-btn {
    font-size: .72rem !important;
    opacity: .7;
    transition: opacity .15s;
}
.clear-btn:hover { opacity: 1; }

/* Speaker grid */
.speaker-info {
    font-size: .75rem; color: #71717a; padding: 4px 0;
}
"""

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.zinc,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#09090b",
    body_background_fill_dark="#09090b",
    block_background_fill="#18181b",
    block_background_fill_dark="#18181b",
    block_border_color="#27272a",
    block_border_color_dark="#27272a",
    block_label_text_color="#a1a1aa",
    block_label_text_color_dark="#a1a1aa",
    block_title_text_color="#f4f4f5",
    block_title_text_color_dark="#f4f4f5",
    body_text_color="#e4e4e7",
    body_text_color_dark="#e4e4e7",
    body_text_color_subdued="#71717a",
    body_text_color_subdued_dark="#71717a",
    input_background_fill="#09090b",
    input_background_fill_dark="#09090b",
    input_border_color="#27272a",
    input_border_color_dark="#27272a",
    button_primary_background_fill="#818cf8",
    button_primary_background_fill_dark="#818cf8",
    button_primary_text_color="#09090b",
    button_primary_text_color_dark="#09090b",
    button_primary_background_fill_hover="#a5b4fc",
    button_primary_background_fill_hover_dark="#a5b4fc",
    button_secondary_background_fill="#27272a",
    button_secondary_background_fill_dark="#27272a",
    button_secondary_text_color="#e4e4e7",
    button_secondary_text_color_dark="#e4e4e7",
    button_secondary_border_color="#3f3f46",
    button_secondary_border_color_dark="#3f3f46",
    border_color_primary="#27272a",
    border_color_primary_dark="#27272a",
    color_accent="#818cf8",
    shadow_drop="none",
    shadow_drop_lg="none",
)

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
LANGUAGES = ["Auto", "English", "Chinese", "Japanese", "Korean", "German",
             "French", "Russian", "Portuguese", "Spanish", "Italian"]

MODEL_SIZES = ["1.7B", "0.6B"]

WHISPER_MODELS = [
    ("tiny (fastest)", "tiny"),
    ("base (default)", "base"),
    ("small", "small"),
    ("medium", "medium"),
    ("large (best)", "large"),
    ("turbo", "turbo"),
]

WHISPER_LANGS = [
    ("Auto-detect", ""),
    ("English", "en"),
    ("Chinese", "zh"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("German", "de"),
    ("French", "fr"),
    ("Russian", "ru"),
    ("Portuguese", "pt"),
    ("Spanish", "es"),
    ("Italian", "it"),
]

DELETE_JS = """
function deleteHistoryItem(fname) {
    const el = document.querySelector('#delete-trigger textarea, #delete-trigger input');
    if (!el) return;
    const setter = Object.getOwnPropertyDescriptor(
        Object.getPrototypeOf(el), 'value'
    ).set;
    setter.call(el, fname);
    el.dispatchEvent(new Event('input', {bubbles: true}));
}
"""

with gr.Blocks(title="VocalFlow") as app:

    with gr.Row():
        # -- Sidebar --
        with gr.Column(scale=1, min_width=160, elem_id="sidebar-col"):
            gr.HTML('<div class="brand-header"><h1>VocalFlow</h1><p>TTS &amp; Transcription</p></div>')
            history_html = gr.HTML(
                value=render_history_html,
                elem_classes=["history-panel"],
            )
            # Hidden textbox used by JS to trigger per-item delete
            delete_trigger = gr.Textbox(visible=False, elem_id="delete-trigger")
            with gr.Row(elem_classes=["sidebar-actions"]):
                clear_btn = gr.Button(
                    "Clear History",
                    variant="stop",
                    size="sm",
                    elem_classes=["clear-btn"],
                )
                unload_btn = gr.Button(
                    "Unload Models",
                    variant="secondary",
                    size="sm",
                    elem_classes=["clear-btn"],
                )

        # -- Main content --
        with gr.Column(scale=3):
            with gr.Tabs():
                # ===== Voice Clone Tab =====
                with gr.Tab("Voice Clone"):
                    gr.Markdown("### Clone a Voice")
                    with gr.Group():
                        vc_text = gr.Textbox(
                            label="Text to Speak",
                            placeholder="Enter the text you want to convert to speech...",
                            lines=4,
                        )
                        with gr.Row():
                            vc_language = gr.Dropdown(
                                choices=LANGUAGES,
                                value="English",
                                label="Language",
                            )
                            vc_model_size = gr.Dropdown(
                                choices=MODEL_SIZES,
                                value="1.7B",
                                label="Model Size",
                            )
                    with gr.Group():
                        vc_ref_audio = gr.Audio(
                            label="Reference Audio (3-10s recommended)",
                            type="filepath",
                            sources=["upload"],
                        )
                        vc_ref_text = gr.Textbox(
                            label="Transcript (recommended for best quality)",
                            placeholder="What the speaker says in the uploaded audio...",
                            lines=2,
                        )
                    vc_btn = gr.Button("Generate Speech", variant="primary", size="lg")

                    vc_audio_out = gr.Audio(label="Generated Speech", type="filepath", interactive=False)
                    vc_file_out = gr.File(label="Download", interactive=False, visible=False)

                    # -- Save / Load Voice sub-section --
                    with gr.Accordion("Save & Load Voice Prompts", open=False):
                        gr.Markdown("Save a reference voice as a reusable `.pt` file, then load it later without re-uploading audio.")
                        with gr.Row():
                            with gr.Column():
                                sv_ref_audio = gr.Audio(
                                    label="Reference Audio",
                                    type="filepath",
                                    sources=["upload"],
                                )
                                sv_ref_text = gr.Textbox(
                                    label="Transcript (optional)",
                                    lines=2,
                                )
                                sv_model_size = gr.Dropdown(
                                    choices=MODEL_SIZES,
                                    value="1.7B",
                                    label="Model Size",
                                )
                                sv_btn = gr.Button("Save Voice", variant="secondary")
                                sv_file_out = gr.File(label="Voice File", interactive=False)
                                sv_status = gr.Textbox(label="Status", interactive=False)

                            with gr.Column():
                                lv_file = gr.File(
                                    label="Upload Saved Voice (.pt)",
                                    file_types=[".pt"],
                                )
                                lv_text = gr.Textbox(
                                    label="Text to Speak",
                                    placeholder="Enter text to generate with saved voice...",
                                    lines=3,
                                )
                                lv_language = gr.Dropdown(
                                    choices=LANGUAGES,
                                    value="English",
                                    label="Language",
                                )
                                lv_model_size = gr.Dropdown(
                                    choices=MODEL_SIZES,
                                    value="1.7B",
                                    label="Model Size",
                                )
                                lv_btn = gr.Button("Generate from Saved Voice", variant="primary")
                                lv_audio_out = gr.Audio(label="Output", type="filepath", interactive=False)
                                lv_file_out = gr.File(label="Download", interactive=False, visible=False)

                # ===== Custom Voice Tab =====
                with gr.Tab("Custom Voice"):
                    gr.Markdown("### Preset Speakers with Instruction Control")
                    gr.HTML('<div class="speaker-info">9 premium voices. The 1.7B model supports instruction control for emotion, tone, and style. The 0.6B model supports speakers only (no instructions).</div>')
                    with gr.Group():
                        cv_text = gr.Textbox(
                            label="Text to Speak",
                            placeholder="Enter the text you want to convert to speech...",
                            lines=4,
                        )
                        with gr.Row():
                            cv_language = gr.Dropdown(
                                choices=LANGUAGES,
                                value="Auto",
                                label="Language",
                            )
                            cv_speaker = gr.Dropdown(
                                choices=SPEAKERS,
                                value="Vivian",
                                label="Speaker",
                            )
                            cv_model_size = gr.Dropdown(
                                choices=MODEL_SIZES,
                                value="1.7B",
                                label="Model Size",
                            )
                    with gr.Group():
                        cv_instruct = gr.Textbox(
                            label="Instruction (optional, 1.7B only)",
                            placeholder="e.g. Say it in an excited, cheerful tone",
                            lines=2,
                        )
                    cv_btn = gr.Button("Generate Speech", variant="primary", size="lg")

                    cv_audio_out = gr.Audio(label="Generated Speech", type="filepath", interactive=False)
                    cv_file_out = gr.File(label="Download", interactive=False, visible=False)

                # ===== Voice Design Tab =====
                with gr.Tab("Voice Design"):
                    gr.Markdown("### Create a Voice from a Description")
                    gr.HTML('<div class="speaker-info">Describe the voice you want in natural language \u2014 no reference audio needed. Uses the 1.7B VoiceDesign model.</div>')
                    with gr.Group():
                        vd_text = gr.Textbox(
                            label="Text to Speak",
                            placeholder="Enter the text you want to convert to speech...",
                            lines=4,
                            value="It's in the top drawer... wait, it's empty? No way, that's impossible!",
                        )
                        vd_language = gr.Dropdown(
                            choices=LANGUAGES,
                            value="Auto",
                            label="Language",
                        )
                    with gr.Group():
                        vd_instruct = gr.Textbox(
                            label="Voice Description",
                            placeholder="e.g. Male, mid-30s, deep baritone, speaking with calm authority",
                            lines=3,
                            value="Speak in an incredulous tone, with a hint of panic creeping into your voice.",
                        )
                    vd_btn = gr.Button("Generate Speech", variant="primary", size="lg")

                    vd_audio_out = gr.Audio(label="Generated Speech", type="filepath", interactive=False)
                    vd_file_out = gr.File(label="Download", interactive=False, visible=False)

                # ===== Transcribe Tab =====
                with gr.Tab("Transcribe"):
                    gr.Markdown("### Transcribe Audio")
                    with gr.Group():
                        tr_audio = gr.Audio(
                            label="Audio File",
                            type="filepath",
                            sources=["upload"],
                        )
                        tr_model = gr.Dropdown(
                            choices=WHISPER_MODELS,
                            value="base",
                            label="Whisper Model",
                        )
                        tr_lang = gr.Dropdown(
                            choices=WHISPER_LANGS,
                            value="",
                            label="Language (optional)",
                        )
                    tr_btn = gr.Button("Transcribe", variant="primary", size="lg")

                    tr_text_out = gr.Textbox(label="Transcript", lines=10, interactive=False)
                    tr_meta_out = gr.Markdown(elem_classes=["transcript-meta"])
                    tr_file_out = gr.File(label="Download JSON", interactive=False, visible=False)

    # -- Event handlers --

    # Voice Clone
    def on_generate_clone(text, language, ref_audio, ref_text, model_size):
        audio_path, file_path, hist = generate_speech(text, language, ref_audio, ref_text, model_size)
        return (
            audio_path,
            gr.File(value=file_path, visible=True),
            hist,
        )

    vc_btn.click(
        fn=on_generate_clone,
        inputs=[vc_text, vc_language, vc_ref_audio, vc_ref_text, vc_model_size],
        outputs=[vc_audio_out, vc_file_out, history_html],
    )

    # Save voice
    def on_save_voice(ref_audio, ref_text, model_size):
        file_path, status = save_voice_prompt(ref_audio, ref_text, model_size)
        return gr.File(value=file_path), status

    sv_btn.click(
        fn=on_save_voice,
        inputs=[sv_ref_audio, sv_ref_text, sv_model_size],
        outputs=[sv_file_out, sv_status],
    )

    # Load voice + generate
    def on_generate_from_voice(text, language, voice_file, model_size):
        audio_path, file_path, hist = generate_from_voice_prompt(text, language, voice_file, model_size)
        return (
            audio_path,
            gr.File(value=file_path, visible=True),
            hist,
        )

    lv_btn.click(
        fn=on_generate_from_voice,
        inputs=[lv_text, lv_language, lv_file, lv_model_size],
        outputs=[lv_audio_out, lv_file_out, history_html],
    )

    # Custom Voice
    def on_generate_custom(text, language, speaker, instruct, model_size):
        audio_path, file_path, hist = generate_custom_voice(text, language, speaker, instruct, model_size)
        return (
            audio_path,
            gr.File(value=file_path, visible=True),
            hist,
        )

    cv_btn.click(
        fn=on_generate_custom,
        inputs=[cv_text, cv_language, cv_speaker, cv_instruct, cv_model_size],
        outputs=[cv_audio_out, cv_file_out, history_html],
    )

    # Voice Design
    def on_generate_design(text, language, instruct):
        audio_path, file_path, hist = generate_voice_design(text, language, instruct)
        return (
            audio_path,
            gr.File(value=file_path, visible=True),
            hist,
        )

    vd_btn.click(
        fn=on_generate_design,
        inputs=[vd_text, vd_language, vd_instruct],
        outputs=[vd_audio_out, vd_file_out, history_html],
    )

    # Transcribe
    def on_transcribe(audio_file, whisper_model_size, whisper_lang):
        full_text, meta, json_path, hist = transcribe_audio(audio_file, whisper_model_size, whisper_lang)
        return (
            full_text,
            meta,
            gr.File(value=json_path, visible=True),
            hist,
        )

    tr_btn.click(
        fn=on_transcribe,
        inputs=[tr_audio, tr_model, tr_lang],
        outputs=[tr_text_out, tr_meta_out, tr_file_out, history_html],
    )

    # Clear history
    clear_btn.click(
        fn=clear_history_and_files,
        inputs=[],
        outputs=[history_html],
    )

    # Unload models
    def on_unload_models():
        n = unload_all_models()
        if n == 0:
            gr.Info("No models are loaded.")
        else:
            gr.Info(f"Unloaded {n} model(s). GPU memory freed.")
        return render_history_html()

    unload_btn.click(
        fn=on_unload_models,
        inputs=[],
        outputs=[history_html],
    )

    # Per-item delete: JS sets the hidden textbox, which triggers this handler
    delete_trigger.change(
        fn=lambda fname: (delete_history_item(fname), ""),
        inputs=[delete_trigger],
        outputs=[history_html, delete_trigger],
    )


if __name__ == "__main__":
    from main import _launch_kwargs
    app.launch(**_launch_kwargs())
