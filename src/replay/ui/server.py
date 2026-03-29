"""Minimal FastAPI app for the Replay local UI.

Exposes run data over HTTP so the browser-based timeline viewer can
fetch it. Full React frontend shipped in Phase 2 — this file wires
the data layer only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def create_app(db_path: Path | None = None) -> Any:
    """Return a FastAPI app instance wired to the local DuckDB store."""
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as exc:
        raise ImportError(
            "FastAPI is required for the Replay UI. "
            "Install it with: pip install 'replay-agent[ui]'"
        ) from exc

    from replay.config import ReplayConfig
    from replay.storage.duckdb_repo import DuckDBRunRepository, DuckDBStepRepository, open_db
    from replay.storage.reader import get_run_summary

    config = ReplayConfig(**({"db_path": db_path} if db_path else {}))
    conn = open_db(config.db_path)
    run_repo = DuckDBRunRepository(conn)
    step_repo = DuckDBStepRepository(conn)

    app = FastAPI(title="Replay", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/api/runs")
    async def list_runs(
        limit: int = 20,
        name: str | None = None,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        runs = await run_repo.list_runs(limit=limit, name=name, tag=tag)
        summaries = []
        for run in runs:
            summary = await get_run_summary(run, step_repo)
            summaries.append(summary.model_dump())
        return summaries

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        from fastapi import HTTPException

        run = await run_repo.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        steps = await step_repo.get_steps(run_id)
        return {
            "run": run.model_dump(),
            "steps": [s.model_dump() for s in steps],
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
