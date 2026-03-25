from __future__ import annotations

import json
from typing import Optional

import anyio
import typer

from replay.config import ReplayConfig
from replay.storage.duckdb_repo import DuckDBRunRepository, DuckDBStepRepository
from replay.storage.reader import get_full_timeline, get_run_summary
from replay.types import Run, RunSummary, Step


_TYPE_COLORS = {
    "llm_call": typer.colors.BLUE,
    "tool_call": typer.colors.GREEN,
    "mcp_call": typer.colors.MAGENTA,
    "memory_read": typer.colors.CYAN,
    "memory_write": typer.colors.CYAN,
    "agent_decision": typer.colors.YELLOW,
}


def _step_prefix(step: Step) -> str:
    color = _TYPE_COLORS.get(step.type.value, typer.colors.WHITE)
    type_label = typer.style(f"[{step.type.value}]", fg=color)
    error_marker = typer.style(" ✗", fg=typer.colors.RED) if step.error else ""
    cost = f"  ${step.cost_usd:.5f}" if step.cost_usd else ""
    tokens = ""
    if step.tokens_in or step.tokens_out:
        tokens = f"  {step.tokens_in or 0}↑ {step.tokens_out or 0}↓"
    return f"  [{step.index:>3}] {type_label}  {step.duration_ms}ms{cost}{tokens}{error_marker}"


def show_command(
    run_id: str = typer.Argument(..., help="Run ID to inspect"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print full input/output"),
) -> None:
    """Show the full step-by-step detail of a run."""
    config = ReplayConfig()
    run_repo = DuckDBRunRepository(config.db_path)
    step_repo = DuckDBStepRepository(config.db_path)

    async def _fetch() -> tuple[Optional[Run], list[Step], Optional[RunSummary]]:
        run = await run_repo.get_run(run_id)
        if run is None:
            return None, [], None
        steps = await get_full_timeline(run, step_repo)
        summary = await get_run_summary(run, step_repo)
        return run, steps, summary

    run, steps, summary = anyio.from_thread.run_sync(anyio.run, _fetch)  # type: ignore[arg-type]

    if run is None:
        typer.echo(f"Run '{run_id}' not found.", err=True)
        raise typer.Exit(code=1)

    assert summary is not None

    # Header
    status_color = (
        typer.colors.GREEN if summary.status == "completed"
        else typer.colors.RED if summary.status == "failed"
        else typer.colors.YELLOW
    )
    typer.echo(
        f"\n{typer.style(run.name, bold=True)}  "
        f"{typer.style(summary.status, fg=status_color)}\n"
        f"  ID:       {run.id}\n"
        f"  Steps:    {summary.step_count}\n"
        f"  Duration: {summary.duration_ms}ms\n"
        f"  Cost:     {'$' + f'{summary.total_cost_usd:.5f}' if summary.total_cost_usd else 'unknown'}\n"
        f"  Tags:     {', '.join(run.tags) or '—'}\n"
    )

    if not steps:
        typer.echo("  (no steps recorded)")
        return

    typer.echo("Steps:")
    for step in steps:
        typer.echo(_step_prefix(step))
        if step.error:
            typer.echo(f"       error: {step.error}")
        if verbose:
            typer.echo(f"       input:  {json.dumps(step.input, indent=6)}")
            typer.echo(f"       output: {json.dumps(step.output, indent=6)}")
        else:
            # Truncated preview
            in_preview = json.dumps(step.input)[:120]
            out_preview = json.dumps(step.output)[:120]
            typer.echo(f"       in:  {in_preview}")
            typer.echo(f"       out: {out_preview}")
    typer.echo("")
