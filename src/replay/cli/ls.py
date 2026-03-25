from __future__ import annotations

from typing import Optional

import anyio
import typer

from replay.config import ReplayConfig
from replay.storage.duckdb_repo import (
    DuckDBRunRepository,
    DuckDBStepRepository,
    open_db,
)
from replay.storage.reader import get_run_summary
from replay.types import RunSummary


def _fmt_cost(cost: Optional[float]) -> str:
    if cost is None:
        return "unknown"
    return f"${cost:.4f}"


def _fmt_duration(ms: int) -> str:
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms / 1000:.1f}s"


def _fmt_when(summary: RunSummary) -> str:
    from datetime import datetime, timezone

    delta = datetime.now(tz=timezone.utc) - summary.started_at
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def ls_command(
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Filter by agent name"),
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Filter by tag"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum number of runs to show"),
) -> None:
    """List recent runs."""
    config = ReplayConfig()
    conn = open_db(config.db_path)
    run_repo = DuckDBRunRepository(conn)
    step_repo = DuckDBStepRepository(conn)

    async def _fetch() -> list[RunSummary]:
        runs = await run_repo.list_runs(limit=limit, name=name, tag=tag)
        summaries = []
        for run in runs:
            summary = await get_run_summary(run, step_repo)
            if status is None or summary.status == status:
                summaries.append(summary)
        return summaries

    summaries = anyio.run(_fetch)

    if not summaries:
        typer.echo("No runs found.")
        return

    id_w, name_w = 10, 20
    header = (
        f"{'ID':<{id_w}}  {'NAME':<{name_w}}  "
        f"{'STEPS':>5}  {'COST':>10}  {'DURATION':>9}  {'STATUS':<10}  WHEN"
    )
    typer.echo(header)
    typer.echo("-" * len(header))

    for s in summaries:
        typer.echo(
            f"{s.run_id[:id_w]:<{id_w}}  "
            f"{s.name[:name_w]:<{name_w}}  "
            f"{s.step_count:>5}  "
            f"{_fmt_cost(s.total_cost_usd):>10}  "
            f"{_fmt_duration(s.duration_ms):>9}  "
            f"{s.status:<10}  "
            f"{_fmt_when(s)}"
        )
