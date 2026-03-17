import re
import uuid
import json
import time
import torch
import soundfile as sf
from pathlib import Path
from datetime import datetime
from starlette.staticfiles import StaticFiles
from fasthtml.common import *

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "audio"
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
HISTORY_FILE = BASE_DIR / "history.json"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
TRANSCRIPT_DIR.mkdir(exist_ok=True)

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
    """Turn text into a safe filename slug."""
    s = text.strip()[:max_len].lower()
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s_]+', '-', s).strip('-')
    return s or "untitled"


def _find_entry(fname: str) -> dict | None:
    """Look up a history entry by internal filename."""
    for e in _load_history():
        if e.get("file") == fname:
            return e
    return None


# ---------------------------------------------------------------------------
# TTS Model loading (lazy)
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
# Whisper model loading (lazy)
# ---------------------------------------------------------------------------
_whisper_model = None
_whisper_size = None


def get_whisper_model(size: str = "base"):
    global _whisper_model, _whisper_size
    if _whisper_model is None or _whisper_size != size:
        import whisper
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[model] Loading Whisper '{size}' on {device.upper()} ...")
        _whisper_model = whisper.load_model(size, device=device)
        _whisper_size = size
        print("[model] Whisper ready.")
    return _whisper_model


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
html{font-size:15px}
body{font-family:'Inter',system-ui,sans-serif;background:#09090b;color:#e4e4e7;min-height:100vh;line-height:1.6}
a{color:#818cf8;text-decoration:none}
a:hover{color:#a5b4fc}

/* Layout: sidebar + main */
.layout{display:flex;min-height:100vh}

/* Sidebar */
.sidebar{
    width:300px;min-width:300px;
    background:#111113;border-right:1px solid #27272a;
    display:flex;flex-direction:column;
    overflow:hidden;
}
.sidebar-brand{
    padding:20px;border-bottom:1px solid #27272a;
    text-align:center;
}
.sidebar-brand h1{font-size:1.2rem;font-weight:700;color:#f4f4f5;letter-spacing:-.02em}
.sidebar-brand p{color:#52525b;font-size:.72rem;margin-top:2px}
.sidebar-hist{flex:1;overflow-y:auto;padding:12px}
.sidebar-hist::-webkit-scrollbar{width:4px}
.sidebar-hist::-webkit-scrollbar-thumb{background:#27272a;border-radius:4px}
.sidebar-hdr{
    display:flex;align-items:center;justify-content:space-between;
    margin-bottom:10px;padding:0 4px;
}
.sidebar-hdr h3{font-size:.7rem;font-weight:600;color:#71717a;text-transform:uppercase;letter-spacing:.5px;margin:0}
.sidebar-count{font-size:.68rem;color:#52525b}

.si{
    display:flex;align-items:flex-start;gap:10px;
    padding:10px 12px;
    background:#18181b;border:1px solid transparent;border-radius:8px;
    margin-bottom:4px;transition:all .15s;cursor:default;
}
.si:hover{border-color:#27272a;background:#1c1c1f}
.si-icon{
    width:26px;height:26px;border-radius:6px;
    display:flex;align-items:center;justify-content:center;
    font-size:.65rem;font-weight:700;flex-shrink:0;margin-top:1px;
}
.si-icon.tts{background:#1e1b4b;color:#a78bfa}
.si-icon.clone{background:#172554;color:#60a5fa}
.si-icon.design{background:#1e1b4b;color:#a78bfa}
.si-icon.transcribe{background:#052e16;color:#4ade80}
.si-meta{flex:1;min-width:0}
.si-text{font-size:.78rem;color:#d4d4d8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.si-sub{font-size:.65rem;color:#52525b;margin-top:1px}
.si-actions{display:flex;gap:3px;flex-shrink:0;margin-top:2px}
.si-btn{
    padding:3px 8px;border-radius:5px;font-size:.65rem;font-weight:500;
    text-decoration:none;border:1px solid #27272a;background:none;cursor:pointer;
    transition:all .15s;color:#a1a1aa;
}
.si-btn:hover{background:#27272a;color:#f4f4f5}
.si-btn.play{color:#4ade80}
.si-btn.play:hover{background:#052e16;border-color:#166534}
.si-btn.dl{color:#818cf8}
.si-btn.dl:hover{background:#1e1b4b;border-color:#818cf8}
.si-player{overflow:hidden;max-height:0;transition:max-height .25s ease}
.si-player.open{max-height:60px;padding:4px 12px 6px}
.si-player audio{width:100%;height:32px;border-radius:6px}
.sidebar-empty{
    text-align:center;color:#52525b;font-size:.78rem;
    padding:30px 16px;
}

/* Main content */
.main{flex:1;overflow-y:auto;padding:32px 40px;max-width:900px}
.main-hdr{margin-bottom:24px}
.main-hdr h2{font-size:1.4rem;font-weight:700;color:#f4f4f5;letter-spacing:-.01em}
.main-hdr p{color:#71717a;font-size:.85rem;margin-top:2px}

/* Page tabs */
.page-tabs{display:flex;gap:2px;margin-bottom:24px;border-bottom:1px solid #27272a}
.ptab{
    padding:10px 20px;color:#71717a;font-size:.88rem;font-weight:500;
    cursor:pointer;border:none;background:none;
    border-bottom:2px solid transparent;transition:all .15s;
}
.ptab:hover{color:#e4e4e7}
.ptab.on{color:#818cf8;border-bottom-color:#818cf8}
.page-panel{display:none}
.page-panel.active{display:block}

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

/* Voice sub-tabs */
.tabs{display:flex;gap:2px;margin-bottom:14px;border-bottom:1px solid #27272a;padding-bottom:0}
.tb{
    padding:10px 18px;color:#a1a1aa;font-size:.875rem;font-weight:500;
    cursor:pointer;border:none;background:none;
    border-bottom:2px solid transparent;transition:all .15s;
}
.tb:hover{color:#f4f4f5}
.tb.on{color:#818cf8;border-bottom-color:#818cf8}
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

/* Buttons */
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

.btn-transcribe{
    display:block;width:100%;padding:12px;margin-top:8px;
    background:#4ade80;color:#09090b;
    border:none;border-radius:8px;
    font-size:.95rem;font-weight:700;cursor:pointer;
    letter-spacing:.3px;transition:all .15s;
}
.btn-transcribe:hover{background:#86efac}
.btn-transcribe:active{transform:scale(.99)}
.btn-transcribe:disabled{opacity:.35;cursor:not-allowed;transform:none}

/* Result */
#tts-result,#transcribe-result{margin-top:16px}
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

/* Transcript result */
.transcript-box{
    background:#18181b;border:1px solid #27272a;
    border-radius:10px;padding:16px 20px;
    max-height:400px;overflow-y:auto;
}
.transcript-box pre{
    font-family:'Inter',system-ui,sans-serif;
    font-size:.88rem;color:#e4e4e7;
    white-space:pre-wrap;word-wrap:break-word;
    line-height:1.7;
}
.transcript-meta{
    font-size:.75rem;color:#71717a;margin-top:8px;
    padding-top:8px;border-top:1px solid #27272a;
}
.transcript-actions{display:flex;gap:8px;margin-top:10px}
.transcript-actions .dl-link{font-size:.78rem}

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

/* Animation */
.fi{animation:fi .3s ease}
@keyframes fi{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}

/* Responsive */
@media(max-width:768px){
    .layout{flex-direction:column}
    .sidebar{width:100%;min-width:100%;max-height:200px;border-right:none;border-bottom:1px solid #27272a}
    .main{padding:20px}
}
"""

JS = """
function togglePlay(uid, src) {
    var el = document.getElementById('player-' + uid);
    if (el.classList.contains('open')) {
        el.classList.remove('open');
        el.innerHTML = '';
        return;
    }
    document.querySelectorAll('.si-player.open').forEach(function(p) {
        p.classList.remove('open'); p.innerHTML = '';
    });
    el.innerHTML = '<audio src="' + src + '" controls autoplay></audio>';
    el.classList.add('open');
}
function switchVoiceTab(mode) {
    document.querySelectorAll('.vtb').forEach(b => b.classList.remove('on'));
    document.querySelectorAll('.voice-panel').forEach(p => p.classList.remove('active'));
    document.querySelector('[data-vtab="'+mode+'"]').classList.add('on');
    document.getElementById('vpanel-'+mode).classList.add('active');
    document.getElementById('voice_mode').value = mode;
}
function switchPage(page) {
    document.querySelectorAll('.ptab').forEach(b => b.classList.remove('on'));
    document.querySelectorAll('.page-panel').forEach(p => p.classList.remove('active'));
    document.querySelector('[data-page="'+page+'"]').classList.add('on');
    document.getElementById('page-'+page).classList.add('active');
}
function copyTranscript() {
    var el = document.getElementById('transcript-text');
    if (el) {
        navigator.clipboard.writeText(el.innerText);
        var btn = document.getElementById('copy-btn');
        btn.innerText = 'Copied!';
        setTimeout(function(){ btn.innerText = 'Copy'; }, 1500);
    }
}
"""

# ---------------------------------------------------------------------------
# FastHTML app
# ---------------------------------------------------------------------------
app, rt = fast_app(hdrs=[Style(CSS), Script(JS)])
app.mount("/audio", StaticFiles(directory=str(OUTPUT_DIR)), name="audio_static")


# ---------------------------------------------------------------------------
# Sidebar components
# ---------------------------------------------------------------------------
def sidebar_item(entry: dict):
    fname = entry["file"]
    mode = entry.get("mode", "")
    is_transcribe = mode == "transcribe"

    if is_transcribe:
        icon_cls = "si-icon transcribe"
        icon_txt = "T"
        mode_label = "Transcribe"
    elif mode == "clone":
        icon_cls = "si-icon clone"
        icon_txt = "C"
        mode_label = "Clone"
    else:
        icon_cls = "si-icon design"
        icon_txt = "D"
        mode_label = "Design"

    voice = entry.get("voice", "")
    sub = f'{entry.get("time", "")}  ·  {mode_label}'
    lang = entry.get("language", "")
    if lang:
        sub += f"  ·  {lang}"
    if voice:
        sub += f"  ·  {voice[:30]}"

    uid = fname.replace(".", "_")

    if is_transcribe:
        json_name = fname
        exists = (TRANSCRIPT_DIR / json_name).exists()
        if exists:
            actions = Div(
                A(NotStr("&#8681;"), href=f"/transcript/{json_name.removesuffix('.json')}", cls="si-btn dl", title="Download"),
                cls="si-actions",
            )
        else:
            actions = Div()
        return Div(
            Div(icon_txt, cls=icon_cls),
            Div(
                Div(entry.get("text", "—"), cls="si-text"),
                Div(sub, cls="si-sub"),
                cls="si-meta",
            ),
            actions,
            cls="si",
        )

    exists = (OUTPUT_DIR / fname).exists()
    if exists:
        actions = Div(
            Button("Play", onclick=f"togglePlay('{uid}','/audio/{fname}')", cls="si-btn play"),
            A(NotStr("&#8681;"), href=f"/download/{fname.removesuffix('.wav')}", download=f"{_slug(entry.get('text', 'audio'))}.wav", cls="si-btn dl", title="Download"),
            cls="si-actions",
        )
        player = Div(id=f"player-{uid}", cls="si-player")
    else:
        actions = Div()
        player = None

    row = Div(
        Div(icon_txt, cls=icon_cls),
        Div(
            Div(entry.get("text", "—"), cls="si-text"),
            Div(sub, cls="si-sub"),
            cls="si-meta",
        ),
        actions,
        cls="si", id=f"si-{uid}",
    )
    if player:
        return Div(row, player)
    return row


def sidebar_history():
    entries = _load_history()
    if not entries:
        return Div(
            Div(
                Div(H3("History"), cls="sidebar-hdr"),
                Div("No activity yet.", cls="sidebar-empty"),
            ),
            id="sidebar-history",
        )
    items = [sidebar_item(e) for e in entries[:30]]
    return Div(
        Div(
            H3("History"),
            Span(f"{len(entries)}", cls="sidebar-count"),
            cls="sidebar-hdr",
        ),
        *items,
        id="sidebar-history",
    )


# ---------------------------------------------------------------------------
# Page: TTS
# ---------------------------------------------------------------------------
def tts_page():
    form = Form(
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
        Div(
            H3("Voice Source"),
            Input(type="hidden", name="voice_mode", id="voice_mode", value="prompt"),
            Div(
                A("Voice Design", cls="vtb on", data_vtab="prompt", onclick="switchVoiceTab('prompt')"),
                A("Clone from Audio", cls="vtb", data_vtab="audio", onclick="switchVoiceTab('audio')"),
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
                id="vpanel-prompt", cls="voice-panel active tab-panel",
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
                id="vpanel-audio", cls="voice-panel tab-panel",
            ),
            cls="cd",
        ),
        Button("Generate Speech", cls="btn-gen", type="submit"),
        hx_post="/generate",
        hx_target="#tts-result",
        hx_indicator="#tts-spinner",
        hx_encoding="multipart/form-data",
        hx_disabled_elt=".btn-gen",
    )

    spinner = Div(
        Div(Span(cls="sp"), Span("Generating speech...", cls="sp-text"), cls="sp-wrap"),
        id="tts-spinner", cls="htmx-indicator",
    )
    result = Div(id="tts-result")
    return Div(form, spinner, result)


# ---------------------------------------------------------------------------
# Page: Transcribe
# ---------------------------------------------------------------------------
def transcribe_page():
    form = Form(
        Div(
            H3("Transcribe Audio"),
            Div(
                Label("Audio File"),
                Input(type="file", name="audio_file", accept=".wav,.mp3,.flac,.ogg,audio/*", required=True),
                cls="field",
            ),
            Div(
                Label("Whisper Model"),
                Select(
                    Option("tiny (fastest)", value="tiny"),
                    Option("base (default)", value="base", selected=True),
                    Option("small", value="small"),
                    Option("medium", value="medium"),
                    Option("large (best)", value="large"),
                    Option("turbo", value="turbo"),
                    name="whisper_model",
                ),
                cls="field",
            ),
            Div(
                Label("Language (optional)"),
                Select(
                    Option("Auto-detect", value=""),
                    Option("English", value="en"),
                    Option("Chinese", value="zh"),
                    Option("Japanese", value="ja"),
                    Option("Korean", value="ko"),
                    Option("German", value="de"),
                    Option("French", value="fr"),
                    Option("Russian", value="ru"),
                    Option("Portuguese", value="pt"),
                    Option("Spanish", value="es"),
                    Option("Italian", value="it"),
                    name="whisper_lang",
                ),
                cls="field",
            ),
            cls="cd",
        ),
        Button("Transcribe", cls="btn-transcribe", type="submit"),
        hx_post="/transcribe",
        hx_target="#transcribe-result",
        hx_indicator="#transcribe-spinner",
        hx_encoding="multipart/form-data",
        hx_disabled_elt=".btn-transcribe",
    )

    spinner = Div(
        Div(Span(cls="sp"), Span("Transcribing audio...", cls="sp-text"), cls="sp-wrap"),
        id="transcribe-spinner", cls="htmx-indicator",
    )
    result = Div(id="transcribe-result")
    return Div(form, spinner, result)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@rt("/")
def get():
    sidebar = Div(
        Div(H1("VocalFlow"), P("TTS & Transcription"), cls="sidebar-brand"),
        Div(sidebar_history(), cls="sidebar-hist"),
        cls="sidebar",
    )

    main = Div(
        Div(
            A("Speech", cls="ptab on", data_page="tts", onclick="switchPage('tts')"),
            A("Transcribe", cls="ptab", data_page="transcribe", onclick="switchPage('transcribe')"),
            cls="page-tabs",
        ),
        Div(tts_page(), id="page-tts", cls="page-panel active"),
        Div(transcribe_page(), id="page-transcribe", cls="page-panel"),
        cls="main",
    )

    return Title("VocalFlow"), Div(sidebar, main, cls="layout")


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
    pretty_name = f"{_slug(text.strip())}.wav"

    return Div(
        Div(
            P("Generated successfully", cls="status"),
            Audio(src=f"/audio/{out_name}", controls=True, autoplay=True),
            A(f"Download {pretty_name}", href=f"/download/{out_name.removesuffix('.wav')}", download=pretty_name, cls="dl-link"),
            cls="result-ok fi",
        ),
        sidebar_history(),
        Script("document.getElementById('sidebar-history').replaceWith(document.querySelectorAll('#sidebar-history')[1])"),
    )


@rt("/transcribe")
async def post_transcribe(audio_file: UploadFile, whisper_model: str = "base",
                          whisper_lang: str = ""):
    if not audio_file or not audio_file.filename:
        return Div(P("Please upload an audio file."), cls="err")

    ext = Path(audio_file.filename).suffix.lower()
    tmp_name = f"{uuid.uuid4().hex}{ext}"
    tmp_path = UPLOAD_DIR / tmp_name
    content = await audio_file.read()
    tmp_path.write_bytes(content)

    try:
        model = get_whisper_model(whisper_model)
        t0 = time.perf_counter()

        opts = {"word_timestamps": True, "verbose": False}
        if whisper_lang:
            opts["language"] = whisper_lang

        result = model.transcribe(str(tmp_path), **opts)
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
            "model": whisper_model,
            "elapsed": round(elapsed, 1),
            "source": audio_file.filename,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        _add_history(json_name, full_text, "transcribe", detected_lang,
                     f"{whisper_model} · {elapsed:.1f}s")

        tmp_path.unlink(missing_ok=True)
        pretty_json = f"{_slug(full_text)}.json"

        return Div(
            Div(
                P("Transcription complete", cls="status"),
                Div(Pre(full_text, id="transcript-text"), cls="transcript-box"),
                Div(
                    f"{len(words_data)} words · {detected_lang} · {whisper_model} · {elapsed:.1f}s",
                    cls="transcript-meta",
                ),
                Div(
                    Button("Copy", id="copy-btn", onclick="copyTranscript()", cls="si-btn dl"),
                    A("Download JSON", href=f"/transcript/{json_name.removesuffix('.json')}", download=pretty_json, cls="dl-link"),
                    cls="transcript-actions",
                ),
                cls="result-ok fi",
            ),
            sidebar_history(),
            Script("document.getElementById('sidebar-history').replaceWith(document.querySelectorAll('#sidebar-history')[1])"),
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        tmp_path.unlink(missing_ok=True)
        return Div(P(f"Transcription failed: {e}"), cls="err")


@rt("/download/{fid}")
def get_download(fid: str):
    fname = fid + ".wav"
    fpath = OUTPUT_DIR / fname
    if not fpath.exists():
        return Response("File not found", status_code=404)
    entry = _find_entry(fname)
    dl_name = f"{_slug(entry['text'])}.wav" if entry else fname
    return Response(
        content=fpath.read_bytes(),
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{dl_name}"'},
    )


@rt("/transcript/{fid}")
def get_transcript(fid: str):
    fname = fid + ".json"
    fpath = TRANSCRIPT_DIR / fname
    if not fpath.exists():
        return Response("File not found", status_code=404)
    entry = _find_entry(fname)
    dl_name = f"{_slug(entry['text'])}.json" if entry else fname
    return Response(
        content=fpath.read_bytes(),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{dl_name}"'},
    )


@rt("/sidebar")
def get_sidebar():
    return sidebar_history()


if __name__ == "__main__":
    serve()
