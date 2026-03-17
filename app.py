import uuid
import json
import torch
import soundfile as sf
from pathlib import Path
from datetime import datetime
from fasthtml.common import *

# ---------------------------------------------------------------------------
# Paths (absolute — immune to cwd changes from uvicorn reloader)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "audio"
HISTORY_FILE = BASE_DIR / "history.json"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

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


# ---------------------------------------------------------------------------
# Model loading (lazy)
# ---------------------------------------------------------------------------
_base_model = None
_design_model = None


def get_base_model():
    global _base_model
    if _base_model is None:
        from qwen_tts import Qwen3TTSModel
        print("[model] Loading Qwen3-TTS-12Hz-1.7B-Base ...")
        _base_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        print("[model] Base model ready.")
    return _base_model


def get_design_model():
    global _design_model
    if _design_model is None:
        from qwen_tts import Qwen3TTSModel
        print("[model] Loading Qwen3-TTS-12Hz-1.7B-VoiceDesign ...")
        _design_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        print("[model] VoiceDesign model ready.")
    return _design_model


# ---------------------------------------------------------------------------
# CSS — RadioFed palette (zinc + indigo)
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
html{font-size:15px}
body{font-family:'Inter',system-ui,sans-serif;background:#09090b;color:#e4e4e7;min-height:100vh;line-height:1.6}
a{color:#818cf8;text-decoration:none}
a:hover{color:#a5b4fc}

.wrap{max-width:760px;margin:0 auto;padding:24px}

/* Header */
.hdr{padding:32px 0 24px;border-bottom:1px solid #27272a;margin-bottom:24px;text-align:center}
.hdr h1{font-size:1.75rem;font-weight:700;color:#f4f4f5;letter-spacing:-.02em}
.hdr p{color:#71717a;font-size:.9rem;margin-top:2px}

/* Cards */
.cd{background:#18181b;border:1px solid #27272a;border-radius:10px;padding:20px;margin-bottom:16px}
.cd:hover{border-color:#3f3f46}
.cd h3{font-size:.8rem;font-weight:600;color:#a1a1aa;margin-bottom:12px;text-transform:uppercase;letter-spacing:.4px}

/* Form fields */
.field{margin-bottom:14px}
.field:last-child{margin-bottom:0}
label{display:block;font-weight:500;margin-bottom:4px;color:#a1a1aa;font-size:.8rem;text-transform:uppercase;letter-spacing:.4px}
textarea,input[type=text],select{
    width:100%;padding:10px 14px;
    border:1px solid #27272a;border-radius:8px;
    background:#09090b;color:#e4e4e7;
    font-size:.9rem;font-family:inherit;resize:vertical;
    transition:border-color .15s,box-shadow .15s;
}
textarea:focus,input:focus,select:focus{outline:none;border-color:#818cf8;box-shadow:0 0 0 3px rgba(129,140,248,.12)}
select{cursor:pointer}

/* Tabs */
.tabs{display:flex;gap:2px;margin-bottom:14px;border-bottom:1px solid #27272a;padding-bottom:0}
.tb{
    padding:10px 18px;color:#a1a1aa;font-size:.875rem;font-weight:500;
    cursor:pointer;border:none;background:none;
    border-bottom:2px solid transparent;transition:all .15s;
}
.tb:hover{color:#f4f4f5}
.tb.on{color:#818cf8;border-bottom-color:#818cf8}

/* Tab panels (voice source) */
.tab-panel{display:none}
.tab-panel.active{display:block}

/* File input */
input[type=file]{padding:6px;font-size:.85rem;color:#a1a1aa}
input[type=file]::file-selector-button{
    background:#27272a;color:#818cf8;
    border:1px solid #3f3f46;padding:6px 14px;
    border-radius:6px;cursor:pointer;margin-right:8px;
    font-weight:500;font-size:.82rem;transition:background .15s;
}
input[type=file]::file-selector-button:hover{background:#3f3f46}

/* Generate button */
.btn-gen{
    display:block;width:100%;padding:12px;margin-top:8px;
    background:#818cf8;color:#09090b;
    border:none;border-radius:8px;
    font-size:.95rem;font-weight:700;cursor:pointer;
    letter-spacing:.3px;transition:all .15s;
}
.btn-gen:hover{background:#a5b4fc}
.btn-gen:active{transform:scale(.99)}
.btn-gen:disabled{opacity:.35;cursor:not-allowed;transform:none}

/* Result */
#result{margin-top:16px}
.result-ok{
    background:#052e16;border:1px solid #166534;
    border-radius:10px;padding:16px 20px;
}
.result-ok .status{color:#4ade80;font-weight:600;font-size:.85rem;margin-bottom:8px}
audio{width:100%;border-radius:6px;margin:6px 0}
.dl-link{
    display:inline-block;margin-top:6px;
    color:#818cf8;font-size:.82rem;font-weight:500;
    transition:color .15s;
}
.dl-link:hover{color:#a5b4fc}

/* Error */
.err{
    background:#450a0a;border:1px solid #7f1d1d;
    border-radius:10px;padding:14px 18px;
    color:#f87171;font-size:.88rem;
}

/* Spinner */
.htmx-indicator{display:none}
.htmx-request .htmx-indicator,.htmx-request.htmx-indicator{display:block}
.sp-wrap{text-align:center;padding:20px}
.sp{
    display:inline-block;width:16px;height:16px;
    border:2px solid #27272a;border-top-color:#818cf8;
    border-radius:50%;animation:spin .5s linear infinite;
    vertical-align:middle;margin-right:8px;
}
@keyframes spin{to{transform:rotate(360deg)}}
.sp-text{color:#71717a;font-size:.88rem}

/* History */
.hist{margin-top:28px}
.hist-hdr{
    display:flex;align-items:center;justify-content:space-between;
    margin-bottom:14px;
}
.hist-hdr h3{font-size:.8rem;font-weight:600;color:#a1a1aa;text-transform:uppercase;letter-spacing:.4px;margin:0}
.hist-count{font-size:.75rem;color:#71717a}

.hi{
    display:flex;align-items:center;gap:12px;
    padding:12px 14px;
    background:#18181b;border:1px solid #27272a;border-radius:8px;
    margin-bottom:6px;transition:border-color .15s;
}
.hi:hover{border-color:#3f3f46}

.hi-icon{
    width:30px;height:30px;border-radius:6px;
    display:flex;align-items:center;justify-content:center;
    font-size:.75rem;font-weight:700;flex-shrink:0;
}
.hi-icon.clone{background:#172554;color:#60a5fa}
.hi-icon.design{background:#1e1b4b;color:#a78bfa}

.hi-meta{flex:1;min-width:0}
.hi-text{font-size:.82rem;color:#e4e4e7;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hi-sub{font-size:.72rem;color:#71717a;margin-top:1px}

.hi-actions{display:flex;gap:4px;flex-shrink:0}
.hi-btn{
    padding:4px 10px;border-radius:6px;font-size:.72rem;font-weight:500;
    text-decoration:none;border:1px solid #27272a;background:none;cursor:pointer;
    transition:all .15s;
}
.hi-btn.play{color:#4ade80}
.hi-btn.play:hover{background:#052e16;border-color:#166534}
.hi-btn.dl{color:#818cf8}
.hi-btn.dl:hover{background:#1e1b4b;border-color:#818cf8}
.hi-btn.miss{color:#f87171;opacity:.5;cursor:default;border-style:dashed}
.hi-player{overflow:hidden;max-height:0;transition:max-height .25s ease}
.hi-player.open{max-height:60px;padding:6px 14px 8px}
.hi-player audio{width:100%;height:36px;border-radius:6px}

.hist-empty{
    text-align:center;color:#71717a;font-size:.85rem;
    padding:24px;background:#18181b;border-radius:10px;
    border:1px dashed #27272a;
}

/* Badges */
.bg{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.72rem;font-weight:500}
.bg-g{background:#052e16;color:#4ade80}
.bg-b{background:#172554;color:#60a5fa}
.bg-y{background:#422006;color:#fbbf24}

/* Animation */
.fi{animation:fi .3s ease}
@keyframes fi{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
"""

JS = """
function togglePlay(uid, src) {
    var el = document.getElementById('player-' + uid);
    if (el.classList.contains('open')) {
        el.classList.remove('open');
        el.innerHTML = '';
        return;
    }
    document.querySelectorAll('.hi-player.open').forEach(function(p) {
        p.classList.remove('open'); p.innerHTML = '';
    });
    el.innerHTML = '<audio src="' + src + '" controls autoplay></audio>';
    el.classList.add('open');
}
function switchTab(mode) {
    document.querySelectorAll('.vtb').forEach(b => b.classList.remove('on'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelector('[data-tab="'+mode+'"]').classList.add('on');
    document.getElementById('panel-'+mode).classList.add('active');
    document.getElementById('voice_mode').value = mode;
}
"""

# ---------------------------------------------------------------------------
# FastHTML app
# ---------------------------------------------------------------------------
app, rt = fast_app(hdrs=[Style(CSS), Script(JS)], static_path=str(BASE_DIR))


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def history_item(entry: dict):
    fname = entry["file"]
    exists = (OUTPUT_DIR / fname).exists()
    is_clone = entry.get("mode") == "clone"
    icon_cls = "hi-icon clone" if is_clone else "hi-icon design"
    icon_txt = "C" if is_clone else "D"
    mode_label = "Clone" if is_clone else "Design"
    voice = entry.get("voice", "")
    sub = f'{entry.get("time", "")}  ·  {mode_label}  ·  {entry.get("language", "")}'
    if voice:
        sub += f"  ·  {voice[:35]}"

    uid = fname.replace(".", "_")
    if exists:
        actions = Div(
            Button("Play", onclick=f"togglePlay('{uid}','/audio/{fname}')", cls="hi-btn play"),
            A("Download", href=f"/dl/{fname}", download=fname, cls="hi-btn dl"),
            cls="hi-actions",
        )
        player = Div(id=f"player-{uid}", cls="hi-player")
    else:
        actions = Div(Span("Missing", cls="hi-btn miss"), cls="hi-actions")
        player = None

    children = [
        Div(icon_txt, cls=icon_cls),
        Div(
            Div(entry.get("text", "—"), cls="hi-text"),
            Div(sub, cls="hi-sub"),
            cls="hi-meta",
        ),
        actions,
    ]
    row = Div(*children, cls="hi", id=f"hi-{uid}")
    if player:
        return Div(row, player)
    return row


def history_section():
    entries = _load_history()
    if not entries:
        return Div(
            Div(
                Div(H3("History"), cls="hist-hdr"),
                Div("No generations yet. Create your first one above.", cls="hist-empty"),
            ),
            cls="hist", id="history",
        )
    items = [history_item(e) for e in entries[:20]]
    return Div(
        Div(
            Div(
                H3("History"),
                Span(f"{len(entries)} generation{'s' if len(entries) != 1 else ''}", cls="hist-count"),
                cls="hist-hdr",
            ),
            *items,
        ),
        cls="hist", id="history",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@rt("/")
def get():
    form = Form(
        # Text input
        Div(
            H3("Input"),
            Div(
                Label("Text to Speak"),
                Textarea(
                    name="text", rows=4,
                    placeholder="Enter the text you want to convert to speech...",
                    required=True,
                ),
                cls="field",
            ),
            Div(
                Label("Language"),
                Select(
                    Option("Auto", value="Auto"),
                    Option("English", value="English", selected=True),
                    Option("Chinese", value="Chinese"),
                    Option("Japanese", value="Japanese"),
                    Option("Korean", value="Korean"),
                    Option("German", value="German"),
                    Option("French", value="French"),
                    Option("Russian", value="Russian"),
                    Option("Portuguese", value="Portuguese"),
                    Option("Spanish", value="Spanish"),
                    Option("Italian", value="Italian"),
                    name="language",
                ),
                cls="field",
            ),
            cls="cd",
        ),
        # Voice source
        Div(
            H3("Voice Source"),
            Input(type="hidden", name="voice_mode", id="voice_mode", value="prompt"),
            Div(
                A("Voice Design", cls="vtb on", data_tab="prompt", onclick="switchTab('prompt')"),
                A("Clone from Audio", cls="vtb", data_tab="audio", onclick="switchTab('audio')"),
                cls="tabs",
            ),
            Div(
                Div(
                    Label("Describe the voice"),
                    Textarea(
                        name="voice_prompt", rows=3,
                        placeholder="e.g. Young female voice, warm and cheerful, slight British accent...",
                    ),
                    cls="field",
                ),
                id="panel-prompt", cls="tab-panel active",
            ),
            Div(
                Div(
                    Label("Reference Audio File"),
                    Input(type="file", name="ref_audio", accept=".wav,.mp3,.flac,.ogg,audio/*"),
                    cls="field",
                ),
                Div(
                    Label("Transcript (recommended)"),
                    Textarea(
                        name="ref_text", rows=2,
                        placeholder="What the speaker says in the uploaded audio...",
                    ),
                    cls="field",
                ),
                id="panel-audio", cls="tab-panel",
            ),
            cls="cd",
        ),
        Button("Generate Speech", cls="btn-gen", type="submit"),
        hx_post="/generate",
        hx_target="#result",
        hx_indicator="#spinner",
        hx_encoding="multipart/form-data",
        hx_disabled_elt=".btn-gen",
    )

    spinner = Div(
        Div(Span(cls="sp"), Span("Generating speech...", cls="sp-text"), cls="sp-wrap"),
        id="spinner", cls="htmx-indicator",
    )
    result = Div(id="result")

    return Title("TTS Portal"), Div(
        Div(H1("TTS Portal"), P("Text-to-Speech · Voice Design & Cloning"), cls="hdr"),
        form,
        spinner,
        result,
        history_section(),
        cls="wrap",
    )


@rt("/generate")
async def post(text: str, language: str, voice_mode: str,
               voice_prompt: str = "", ref_text: str = "",
               ref_audio: UploadFile = None):
    if not text.strip():
        return Div(P("Please enter some text to speak."), cls="err")

    out_name = f"{uuid.uuid4().hex}.wav"
    out_path = OUTPUT_DIR / out_name
    mode_label = "design"
    voice_desc = ""

    try:
        if voice_mode == "audio" and ref_audio and ref_audio.filename:
            mode_label = "clone"
            ext = Path(ref_audio.filename).suffix.lower()
            tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
            content = await ref_audio.read()
            tmp_path.write_bytes(content)
            voice_desc = f"Cloned from {ref_audio.filename}"

            model = get_base_model()
            wavs, sr = model.generate_voice_clone(
                text=text.strip(),
                language=language,
                ref_audio=str(tmp_path),
                ref_text=ref_text.strip() if ref_text.strip() else None,
                x_vector_only_mode=not bool(ref_text.strip()),
                max_new_tokens=2048,
            )
            sf.write(str(out_path), wavs[0], sr)
            tmp_path.unlink(missing_ok=True)
        else:
            voice_desc = voice_prompt.strip() or "Natural, clear adult voice"
            model = get_design_model()
            wavs, sr = model.generate_voice_design(
                text=text.strip(),
                language=language,
                instruct=voice_desc,
                max_new_tokens=2048,
            )
            sf.write(str(out_path), wavs[0], sr)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return Div(P(f"Generation failed: {e}"), cls="err")

    if not out_path.exists():
        return Div(P("Output file was not created. Please try again."), cls="err")

    _add_history(out_name, text.strip(), mode_label, language, voice_desc)

    return Div(
        Div(
            P("Generated successfully", cls="status"),
            Audio(src=f"/audio/{out_name}", controls=True, autoplay=True),
            A(f"Download {out_name}", href=f"/dl/{out_name}", download=out_name, cls="dl-link"),
            cls="result-ok fi",
        ),
    )


# ---------------------------------------------------------------------------
# Download route (audio playback is handled by FastHTML's built-in static
# handler which serves /audio/xxx.wav from BASE_DIR/audio/)
# ---------------------------------------------------------------------------
@rt("/dl/{fname}")
def get_download(fname: str):
    fpath = OUTPUT_DIR / fname
    if not fpath.exists():
        return Response("File not found", status_code=404)
    return Response(
        content=fpath.read_bytes(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{fname}"',
            "Content-Length": str(fpath.stat().st_size),
        },
    )


@rt("/history")
def get_history():
    return history_section()


serve()
