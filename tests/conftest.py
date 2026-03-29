from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pytest

from replay.storage.repository import RunRepository, StepRepository
from replay.types import Run, Step


# ─────────────────────────────────────────────────────────────────────────────
# In-memory repositories — used by all unit tests.
# Never hit DuckDB. DuckDB tests live in tests/integration/.
# ─────────────────────────────────────────────────────────────────────────────


class InMemoryRunRepository(RunRepository):
    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}

    async def save_run(self, run: Run) -> None:
        self._runs[run.id] = run

    async def get_run(self, run_id: str) -> Optional[Run]:
        return self._runs.get(run_id)

    async def update_run_ended(
        self, run_id: str, ended_at: object, error: Optional[str] = None
    ) -> None:
        run = self._runs.get(run_id)
        if run is not None:
            self._runs[run_id] = run.model_copy(update={"ended_at": ended_at, "error": error})

    async def list_runs(
        self,
        limit: int = 20,
        name: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> list[Run]:
        runs = list(self._runs.values())
        if name is not None:
            runs = [r for r in runs if name.lower() in r.name.lower()]
        if tag is not None:
            runs = [r for r in runs if tag in r.tags]
        runs.sort(key=lambda r: r.started_at, reverse=True)
        return runs[:limit]


class InMemoryStepRepository(StepRepository):
    def __init__(self) -> None:
        self._steps: list[Step] = []

    async def save_step(self, step: Step) -> None:
        self._steps.append(step)

    async def get_steps(self, run_id: str) -> list[Step]:
        return sorted(
            [s for s in self._steps if s.run_id == run_id],
            key=lambda s: s.index,
        )

    async def get_step(self, run_id: str, index: int) -> Optional[Step]:
        for step in self._steps:
            if step.run_id == run_id and step.index == index:
                return step
        return None

    async def get_steps_up_to(self, run_id: str, max_index: int) -> list[Step]:
        return sorted(
            [s for s in self._steps if s.run_id == run_id and s.index <= max_index],
            key=lambda s: s.index,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def run_repo() -> InMemoryRunRepository:
    return InMemoryRunRepository()


@pytest.fixture
def step_repo() -> InMemoryStepRepository:
    return InMemoryStepRepository()


@pytest.fixture
def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)
