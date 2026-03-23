def _launch_kwargs():
    from app import CUSTOM_CSS, DELETE_JS, OUTPUT_DIR, TRANSCRIPT_DIR, theme
    return dict(
        server_name="127.0.0.1",
        server_port=5001,
        allowed_paths=[str(OUTPUT_DIR), str(TRANSCRIPT_DIR)],
        css=CUSTOM_CSS,
        js=DELETE_JS,
        theme=theme,
    )


def main():
    """Entry point for `vocalflow` CLI command."""
    from app import app
    app.launch(**_launch_kwargs())


def transcribe_cli():
    """Entry point for `vocalflow-transcribe` CLI command."""
    from transcribe import main as _main
    _main()


if __name__ == "__main__":
    main()
