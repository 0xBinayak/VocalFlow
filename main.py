def main():
    """Entry point for `vocalflow` CLI command."""
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)


def transcribe_cli():
    """Entry point for `vocalflow-transcribe` CLI command."""
    from transcribe import main as _main
    _main()


if __name__ == "__main__":
    main()
