from __future__ import annotations

from datetime import datetime, timezone

import pytest

from replay.storage.reader import get_full_timeline, get_run_summary
from replay.types import Run, Step, StepType
from tests.conftest import InMemoryRunRepository, InMemoryStepRepository


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _step(run_id: str, index: int, cost: float | None = None) -> Step:
    return Step(
        id=f"s-{run_id}-{index}",
        run_id=run_id,
        index=index,
        type=StepType.LLM_CALL,
        started_at=_now(),
        ended_at=_now(),
        duration_ms=100,
        input={},
        output={},
        cost_usd=cost,
        model="gpt-4o" if cost else None,
        tokens_in=100 if cost else None,
        tokens_out=50 if cost else None,
    )


class TestGetFullTimeline:
    @pytest.mark.anyio
    async def test_root_run_returns_own_steps(
        self,
        step_repo: InMemoryStepRepository,
    ) -> None:
        run = Run(id="r1", name="root", started_at=_now())
        await step_repo.save_step(_step("r1", 0))
        await step_repo.save_step(_step("r1", 1))

        steps = await get_full_timeline(run, step_repo)
        assert len(steps) == 2
        assert [s.index for s in steps] == [0, 1]

    @pytest.mark.anyio
    async def test_fork_combines_parent_and_own_steps(
        self,
        step_repo: InMemoryStepRepository,
    ) -> None:
        # Parent has steps 0, 1, 2, 3
        for i in range(4):
            await step_repo.save_step(_step("parent", i))

        # Fork from step 1 — gets parent steps 0..1 + fork's own steps
        fork_run = Run(
            id="fork1",
            name="fork",
            started_at=_now(),
            parent_run_id="parent",
            fork_at_step=1,
        )
        await step_repo.save_step(_step("fork1", 0))
        await step_repo.save_step(_step("fork1", 1))

        steps = await get_full_timeline(fork_run, step_repo)
        # parent steps 0..1 (2 steps) + fork steps 0..1 (2 steps) = 4 total
        assert len(steps) == 4


class TestGetRunSummary:
    @pytest.mark.anyio
    async def test_status_completed_when_no_errors(
        self,
        step_repo: InMemoryStepRepository,
    ) -> None:
        run = Run(id="r1", name="test", started_at=_now(), ended_at=_now())
        await step_repo.save_step(_step("r1", 0))

        summary = await get_run_summary(run, step_repo)
        assert summary.status == "completed"

    @pytest.mark.anyio
    async def test_status_running_when_not_ended(
        self,
        step_repo: InMemoryStepRepository,
    ) -> None:
        run = Run(id="r1", name="test", started_at=_now())  # no ended_at
        summary = await get_run_summary(run, step_repo)
        assert summary.status == "running"

    @pytest.mark.anyio
    async def test_status_failed_when_step_has_error(
        self,
        step_repo: InMemoryStepRepository,
    ) -> None:
        run = Run(id="r1", name="test", started_at=_now(), ended_at=_now())
        step = Step(
            id="s1",
            run_id="r1",
            index=0,
            type=StepType.LLM_CALL,
            started_at=_now(),
            ended_at=_now(),
            duration_ms=10,
            input={},
            output={},
            error="ValueError: something went wrong",
        )
        await step_repo.save_step(step)

        summary = await get_run_summary(run, step_repo)
        assert summary.status == "failed"

    @pytest.mark.anyio
    async def test_total_cost_is_none_when_no_costs(
        self,
        step_repo: InMemoryStepRepository,
    ) -> None:
        run = Run(id="r1", name="test", started_at=_now(), ended_at=_now())
        await step_repo.save_step(_step("r1", 0, cost=None))

        summary = await get_run_summary(run, step_repo)
        assert summary.total_cost_usd is None

    @pytest.mark.anyio
    async def test_total_cost_sums_steps(
        self,
        step_repo: InMemoryStepRepository,
    ) -> None:
        run = Run(id="r1", name="test", started_at=_now(), ended_at=_now())
        s1 = _step("r1", 0, cost=0.001)
        s2 = _step("r1", 1, cost=0.002)
        await step_repo.save_step(s1)
        await step_repo.save_step(s2)

        summary = await get_run_summary(run, step_repo)
        assert summary.total_cost_usd is not None
        assert abs(summary.total_cost_usd - 0.003) < 1e-9
