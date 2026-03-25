from __future__ import annotations

from typing import Optional

from replay.config import ReplayConfig
from replay.core.circuit_breaker import CircuitBreaker
from replay.core.recorder import Recorder
from replay.storage.duckdb_repo import DuckDBRunRepository, DuckDBStepRepository
from replay.strategies.redaction import (
    FieldRedaction,
    NoOpRedaction,
    PIIRedaction,
    RedactionStrategy,
)


class RecorderFactory:
    """Constructs a fully-wired Recorder for each traced function entry.

    The decorator never instantiates Recorder directly — it calls this factory.
    Swapping from DuckDB to Postgres (Phase 4) only requires changing this class.
    """

    def __init__(self, config: ReplayConfig) -> None:
        self._config = config
        # Shared repositories across all runs in this process.
        # DuckDB allows one writer at a time — reusing the connection is required.
        self._run_repo = DuckDBRunRepository(config.db_path)
        self._step_repo = DuckDBStepRepository(config.db_path)
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
