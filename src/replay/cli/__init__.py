from __future__ import annotations

import typer

from replay.cli.ls import ls_command
from replay.cli.show import show_command

app = typer.Typer(
    name="replay",
    help="Time-travel debugger for AI agents.",
    no_args_is_help=True,
)

app.command("ls")(ls_command)
app.command("show")(show_command)
