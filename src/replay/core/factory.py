from __future__ import annotations

from typing import Optional

from replay.config import ReplayConfig
from replay.core.circuit_breaker import CircuitBreaker
from replay.core.recorder import Recorder
from replay.storage.duckdb_repo import (
    DuckDBRunRepository,
    DuckDBStepRepository,
    open_db,
)
from replay.storage.repository import RunRepository, StepRepository
from replay.strategies.redaction import (
    FieldRedaction,
    NoOpRedaction,
    PIIRedaction,
    RedactionStrategy,
)


class RecorderFactory:
    """Constructs a fully-wired Recorder for each traced function entry.

    The decorator never instantiates Recorder directly — it calls this factory.
    Swapping DuckDB for Postgres (Phase 4) only requires changing this class.

    Repos can be injected — used by tests to pass in-memory implementations
    without any filesystem access.
    """

    def __init__(
        self,
        config: ReplayConfig,
        *,
        run_repo: Optional[RunRepository] = None,
        step_repo: Optional[StepRepository] = None,
    ) -> None:
        self._config = config

        if run_repo is not None and step_repo is not None:
            # Injected — used by tests with in-memory repos.
            self._run_repo = run_repo
            self._step_repo = step_repo
        else:
            # Production path — one shared connection, both repos on it.
            conn = open_db(config.db_path)
            self._run_repo = DuckDBRunRepository(conn)
            self._step_repo = DuckDBStepRepository(conn)

        self._redaction = self._build_redaction(config)

    def create(
        self,
        name: str,
        tags: list[str],
        parent_run_id: Optional[str] = None,
        fork_at_step: Optional[int] = None,
    ) -> Recorder:
        return Recorder(
            name=name,
            tags=tags,
            run_repo=self._run_repo,
            step_repo=self._step_repo,
            redaction=self._redaction,
            circuit_breaker=CircuitBreaker(
                failure_threshold=self._config.circuit_breaker_threshold,
            ),
            parent_run_id=parent_run_id,
            fork_at_step=fork_at_step,
        )

    @staticmethod
    def _build_redaction(config: ReplayConfig) -> RedactionStrategy:
        if config.redact_pii:
            return PIIRedaction()
        if config.redact_fields:
            return FieldRedaction(config.redact_fields)
        return NoOpRedaction()
