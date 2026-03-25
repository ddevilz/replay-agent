from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from replay.storage.duckdb_repo import (
    DuckDBRunRepository,
    DuckDBStepRepository,
    open_db,
)
from replay.types import Run, Step, StepType


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


@pytest.fixture
def conn(tmp_path: Path) -> object:
    return open_db(tmp_path / "test_runs.db")


@pytest.fixture
def run_repo(conn: object) -> DuckDBRunRepository:
    return DuckDBRunRepository(conn)  # type: ignore[arg-type]


@pytest.fixture
def step_repo(conn: object) -> DuckDBStepRepository:
    return DuckDBStepRepository(conn)  # type: ignore[arg-type]


@pytest.mark.integration
class TestDuckDBRunRepository:
    @pytest.mark.anyio
    async def test_save_and_retrieve_run(self, run_repo: DuckDBRunRepository) -> None:
        run = Run(id="r1", name="booking-agent", started_at=_now(), tags=["prod"])
        await run_repo.save_run(run)

        fetched = await run_repo.get_run("r1")
        assert fetched is not None
        assert fetched.id == "r1"
        assert fetched.name == "booking-agent"
        assert "prod" in fetched.tags

    @pytest.mark.anyio
    async def test_get_run_returns_none_for_missing(
        self, run_repo: DuckDBRunRepository
    ) -> None:
        result = await run_repo.get_run("nonexistent")
        assert result is None

    @pytest.mark.anyio
    async def test_update_run_ended(self, run_repo: DuckDBRunRepository) -> None:
        run = Run(id="r1", name="test", started_at=_now())
        await run_repo.save_run(run)
        await run_repo.update_run_ended("r1", _now())

        fetched = await run_repo.get_run("r1")
        assert fetched is not None
        assert fetched.ended_at is not None

    @pytest.mark.anyio
    async def test_list_runs_newest_first(self, run_repo: DuckDBRunRepository) -> None:
        import anyio

        await run_repo.save_run(Run(id="r1", name="old", started_at=_now()))
        await anyio.sleep(0.01)
        await run_repo.save_run(Run(id="r2", name="new", started_at=_now()))

        runs = await run_repo.list_runs(limit=10)
        assert runs[0].id == "r2"
        assert runs[1].id == "r1"

    @pytest.mark.anyio
    async def test_list_runs_filter_by_name(self, run_repo: DuckDBRunRepository) -> None:
        await run_repo.save_run(Run(id="r1", name="booking-agent", started_at=_now()))
        await run_repo.save_run(Run(id="r2", name="search-agent", started_at=_now()))

        runs = await run_repo.list_runs(name="booking")
        assert len(runs) == 1
        assert runs[0].id == "r1"

    @pytest.mark.anyio
    async def test_fork_fields_round_trip(self, run_repo: DuckDBRunRepository) -> None:
        run = Run(
            id="fork1",
            name="fork",
            started_at=_now(),
            parent_run_id="parent1",
            fork_at_step=3,
        )
        await run_repo.save_run(run)
        fetched = await run_repo.get_run("fork1")
        assert fetched is not None
        assert fetched.parent_run_id == "parent1"
        assert fetched.fork_at_step == 3


@pytest.mark.integration
class TestDuckDBStepRepository:
    @pytest.mark.anyio
    async def test_save_and_retrieve_step(
        self,
        run_repo: DuckDBRunRepository,
        step_repo: DuckDBStepRepository,
    ) -> None:
        await run_repo.save_run(Run(id="r1", name="test", started_at=_now()))

        step = Step(
            id="s1",
            run_id="r1",
            index=0,
            type=StepType.LLM_CALL,
            started_at=_now(),
            ended_at=_now(),
            duration_ms=200,
            input={"prompt": "hello"},
            output={"text": "world"},
            model="gpt-4o",
            tokens_in=10,
            tokens_out=5,
            cost_usd=0.0001,
        )
        await step_repo.save_step(step)

        steps = await step_repo.get_steps("r1")
        assert len(steps) == 1
        assert steps[0].id == "s1"
        assert steps[0].input == {"prompt": "hello"}
        assert steps[0].cost_usd == pytest.approx(0.0001)

    @pytest.mark.anyio
    async def test_steps_ordered_by_index(
        self,
        run_repo: DuckDBRunRepository,
        step_repo: DuckDBStepRepository,
    ) -> None:
        await run_repo.save_run(Run(id="r1", name="test", started_at=_now()))

        def _s(idx: int) -> Step:
            return Step(
                id=f"s{idx}",
                run_id="r1",
                index=idx,
                type=StepType.TOOL_CALL,
                started_at=_now(),
                ended_at=_now(),
                duration_ms=10,
                input={},
                output={},
            )

        await step_repo.save_step(_s(2))
        await step_repo.save_step(_s(0))
        await step_repo.save_step(_s(1))

        steps = await step_repo.get_steps("r1")
        assert [s.index for s in steps] == [0, 1, 2]

    @pytest.mark.anyio
    async def test_get_steps_up_to(
        self,
        run_repo: DuckDBRunRepository,
        step_repo: DuckDBStepRepository,
    ) -> None:
        await run_repo.save_run(Run(id="r1", name="test", started_at=_now()))

        for i in range(5):
            await step_repo.save_step(
                Step(
                    id=f"s{i}",
                    run_id="r1",
                    index=i,
                    type=StepType.LLM_CALL,
                    started_at=_now(),
                    ended_at=_now(),
                    duration_ms=10,
                    input={},
                    output={},
                )
            )

        steps = await step_repo.get_steps_up_to("r1", max_index=2)
        assert len(steps) == 3
        assert [s.index for s in steps] == [0, 1, 2]

    @pytest.mark.anyio
    async def test_shared_connection_no_lock_error(
        self,
        run_repo: DuckDBRunRepository,
        step_repo: DuckDBStepRepository,
    ) -> None:
        """Run and step writes on the same connection must not deadlock."""
        run = Run(id="r1", name="test", started_at=_now())
        await run_repo.save_run(run)

        step = Step(
            id="s1",
            run_id="r1",
            index=0,
            type=StepType.LLM_CALL,
            started_at=_now(),
            ended_at=_now(),
            duration_ms=5,
            input={},
            output={},
        )
        # Interleaved writes on same connection — must not raise
        await step_repo.save_step(step)
        await run_repo.update_run_ended("r1", _now())

        fetched = await run_repo.get_run("r1")
        assert fetched is not None
        assert fetched.ended_at is not None
