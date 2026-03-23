def main():
    """Entry point for `vocalflow` CLI command."""
    from app import app
    app.launch(server_name="127.0.0.1", server_port=5001)


def transcribe_cli():
    """Entry point for `vocalflow-transcribe` CLI command."""
    from transcribe import main as _main
    _main()


if __name__ == "__main__":
    main()
