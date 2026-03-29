"""replay ui — open the local timeline UI in browser."""
from __future__ import annotations

import typer


def ui_command(
    port: int = typer.Option(4242, "--port", "-p", help="Port to listen on"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser automatically"),
) -> None:
    """Open the local timeline UI in browser."""
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        typer.echo(
            "FastAPI and uvicorn are required for the Replay UI.\n"
            "Install with: pip install 'replay-agent[ui]'",
            err=True,
        )
        raise typer.Exit(code=1)

    from replay.ui.server import create_app

    url = f"http://localhost:{port}"
    typer.echo(f"→ Replay UI at {url}")
    typer.echo("→ Press Ctrl+C to stop")

    if not no_browser:
        import webbrowser
        webbrowser.open(url)

    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=port)
