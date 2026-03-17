def main():
    """Entry point for `vocalflow` CLI command."""
    from app import serve
    serve()


def transcribe_cli():
    """Entry point for `vocalflow-transcribe` CLI command."""
    from transcribe import main as _main
    _main()


if __name__ == "__main__":
    main()
