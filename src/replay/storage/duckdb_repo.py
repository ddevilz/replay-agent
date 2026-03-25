from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import anyio
import duckdb

from replay.storage.repository import RunRepository, StepRepository
from replay.types import Run, Step, StepType

logger = logging.getLogger("replay.storage")

_SCHEMA_VERSION = 1

_DDL = """\
CREATE TABLE IF NOT EXISTS schema_version (
    version  INTEGER PRIMARY KEY,
    applied  TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runs (
    id            VARCHAR PRIMARY KEY,
    name          VARCHAR NOT NULL,
    started_at    TIMESTAMP NOT NULL,
    ended_at      TIMESTAMP,
    tags          JSON     NOT NULL DEFAULT '[]',
    parent_run_id VARCHAR,
    fork_at_step  INTEGER
);

CREATE TABLE IF NOT EXISTS steps (
    id               VARCHAR PRIMARY KEY,
    run_id           VARCHAR NOT NULL REFERENCES runs(id),
    index            INTEGER NOT NULL,
    type             VARCHAR NOT NULL,
    started_at       TIMESTAMP NOT NULL,
    ended_at         TIMESTAMP NOT NULL,
    duration_ms      INTEGER NOT NULL,
    input            JSON NOT NULL,
    output           JSON NOT NULL,
    error            VARCHAR,
    model            VARCHAR,
    tokens_in        INTEGER,
    tokens_out       INTEGER,
    cached_tokens_in INTEGER,
    cost_usd         DOUBLE,
    metadata         JSON NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_steps_run_id    ON steps(run_id);
CREATE INDEX IF NOT EXISTS idx_steps_run_index ON steps(run_id, index);
CREATE INDEX IF NOT EXISTS idx_runs_started    ON runs(started_at DESC);
"""


def open_db(path: Path) -> duckdb.DuckDBPyConnection:
    """Open (or create) the DuckDB database and apply schema.

    Returns the single shared connection. Callers must pass this connection
    to both DuckDBRunRepository and DuckDBStepRepository — DuckDB allows only
    one writer at a time on a local file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    conn.execute(_DDL)
    conn.execute(
        "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
        [_SCHEMA_VERSION],
    )
    return conn


# ──────────────────────────────────────────────────────────────────────────────
# Datetime helpers — DuckDB stores as naive UTC, Python uses aware UTC
# ──────────────────────────────────────────────────────────────────────────────


def _to_naive_utc(dt: datetime) -> datetime:
    """Strip timezone for DuckDB TIMESTAMP storage (always UTC internally)."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _to_aware_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Re-attach UTC timezone when reading back from DuckDB TIMESTAMP."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ──────────────────────────────────────────────────────────────────────────────
# Row mappers — pure functions, no side effects
# ──────────────────────────────────────────────────────────────────────────────


def _row_to_run(row: tuple[Any, ...]) -> Run:
    return Run(
        id=row[0],
        name=row[1],
        started_at=_to_aware_utc(row[2]),  # type: ignore[arg-type]
        ended_at=_to_aware_utc(row[3]),
        tags=json.loads(row[4]) if isinstance(row[4], str) else (row[4] or []),
        parent_run_id=row[5],
        fork_at_step=row[6],
    )


def _row_to_step(row: tuple[Any, ...]) -> Step:
    return Step(
        id=row[0],
        run_id=row[1],
        index=row[2],
        type=StepType(row[3]),
        started_at=_to_aware_utc(row[4]),  # type: ignore[arg-type]
        ended_at=_to_aware_utc(row[5]),  # type: ignore[arg-type]
        duration_ms=row[6],
        input=json.loads(row[7]) if isinstance(row[7], str) else row[7],
        output=json.loads(row[8]) if isinstance(row[8], str) else row[8],
        error=row[9],
        model=row[10],
        tokens_in=row[11],
        tokens_out=row[12],
        cached_tokens_in=row[13],
        cost_usd=row[14],
        metadata=json.loads(row[15]) if isinstance(row[15], str) else (row[15] or {}),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Repositories
# ──────────────────────────────────────────────────────────────────────────────


class DuckDBRunRepository(RunRepository):
    """DuckDB-backed run storage.

    Accepts a shared connection — do not create separate connections per repo.
    DuckDB only allows one writer at a time on a local file.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    # ------------------------------------------------------------------ writes

    async def save_run(self, run: Run) -> None:
        await anyio.to_thread.run_sync(self._save_run_sync, run)

    def _save_run_sync(self, run: Run) -> None:
        self._conn.execute(
            "INSERT INTO runs (id, name, started_at, ended_at, tags, parent_run_id, fork_at_step) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                run.id,
                run.name,
                _to_naive_utc(run.started_at),
                _to_naive_utc(run.ended_at) if run.ended_at else None,
                json.dumps(run.tags),
                run.parent_run_id,
                run.fork_at_step,
            ],
        )

    async def update_run_ended(self, run_id: str, ended_at: datetime) -> None:
        await anyio.to_thread.run_sync(self._update_ended_sync, run_id, ended_at)

    def _update_ended_sync(self, run_id: str, ended_at: datetime) -> None:
        self._conn.execute(
            "UPDATE runs SET ended_at = ? WHERE id = ?",
            [_to_naive_utc(ended_at), run_id],
        )

    # ------------------------------------------------------------------ reads

    async def get_run(self, run_id: str) -> Optional[Run]:
        return await anyio.to_thread.run_sync(self._get_run_sync, run_id)

    def _get_run_sync(self, run_id: str) -> Optional[Run]:
        row = self._conn.execute(
            "SELECT id, name, started_at, ended_at, tags, parent_run_id, fork_at_step "
            "FROM runs WHERE id = ?",
            [run_id],
        ).fetchone()
        return _row_to_run(row) if row else None

    async def list_runs(
        self,
        limit: int = 20,
        name: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> list[Run]:
        return await anyio.to_thread.run_sync(self._list_runs_sync, limit, name, tag)

    def _list_runs_sync(
        self, limit: int, name: Optional[str], tag: Optional[str]
    ) -> list[Run]:
        clauses: list[str] = []
        params: list[Any] = []

        if name is not None:
            clauses.append("name LIKE ?")
            params.append(f"%{name}%")
        if tag is not None:
            clauses.append("json_contains(tags, ?)::BOOLEAN")
            params.append(json.dumps(tag))

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        rows = self._conn.execute(
            f"SELECT id, name, started_at, ended_at, tags, parent_run_id, fork_at_step "
            f"FROM runs {where} ORDER BY started_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [_row_to_run(r) for r in rows]


class DuckDBStepRepository(StepRepository):
    """DuckDB-backed step storage. Append-only — no UPDATE path exists.

    Accepts a shared connection — must be the same connection used by
    DuckDBRunRepository. See open_db().
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    # ------------------------------------------------------------------ writes

    async def save_step(self, step: Step) -> None:
        await anyio.to_thread.run_sync(self._save_step_sync, step)

    def _save_step_sync(self, step: Step) -> None:
        self._conn.execute(
            "INSERT INTO steps ("
            "  id, run_id, index, type, started_at, ended_at, duration_ms,"
            "  input, output, error, model, tokens_in, tokens_out,"
            "  cached_tokens_in, cost_usd, metadata"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                step.id,
                step.run_id,
                step.index,
                step.type.value,
                _to_naive_utc(step.started_at),
                _to_naive_utc(step.ended_at),
                step.duration_ms,
                json.dumps(step.input),
                json.dumps(step.output),
                step.error,
                step.model,
                step.tokens_in,
                step.tokens_out,
                step.cached_tokens_in,
                step.cost_usd,
                json.dumps(step.metadata),
            ],
        )

    # ------------------------------------------------------------------ reads

    async def get_steps(self, run_id: str) -> list[Step]:
        return await anyio.to_thread.run_sync(self._get_steps_sync, run_id)

    def _get_steps_sync(self, run_id: str) -> list[Step]:
        rows = self._conn.execute(
            "SELECT id, run_id, index, type, started_at, ended_at, duration_ms, "
            "input, output, error, model, tokens_in, tokens_out, "
            "cached_tokens_in, cost_usd, metadata "
            "FROM steps WHERE run_id = ? ORDER BY index",
            [run_id],
        ).fetchall()
        return [_row_to_step(r) for r in rows]

    async def get_step(self, run_id: str, index: int) -> Optional[Step]:
        return await anyio.to_thread.run_sync(self._get_step_sync, run_id, index)

    def _get_step_sync(self, run_id: str, index: int) -> Optional[Step]:
        row = self._conn.execute(
            "SELECT id, run_id, index, type, started_at, ended_at, duration_ms, "
            "input, output, error, model, tokens_in, tokens_out, "
            "cached_tokens_in, cost_usd, metadata "
            "FROM steps WHERE run_id = ? AND index = ?",
            [run_id, index],
        ).fetchone()
        return _row_to_step(row) if row else None

    async def get_steps_up_to(self, run_id: str, max_index: int) -> list[Step]:
        return await anyio.to_thread.run_sync(
            self._get_steps_up_to_sync, run_id, max_index
        )

    def _get_steps_up_to_sync(self, run_id: str, max_index: int) -> list[Step]:
        rows = self._conn.execute(
            "SELECT id, run_id, index, type, started_at, ended_at, duration_ms, "
            "input, output, error, model, tokens_in, tokens_out, "
            "cached_tokens_in, cost_usd, metadata "
            "FROM steps WHERE run_id = ? AND index <= ? ORDER BY index",
            [run_id, max_index],
        ).fetchall()
        return [_row_to_step(r) for r in rows]
