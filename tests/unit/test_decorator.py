from __future__ import annotations

import pytest

from replay.config import ReplayConfig
from replay.core.context import get_recorder
from replay.core.decorator import _set_factory, trace
from replay.core.factory import RecorderFactory
from replay.types import StepType
from tests.conftest import InMemoryRunRepository, InMemoryStepRepository


def _make_factory(
    run_repo: InMemoryRunRepository,
    step_repo: InMemoryStepRepository,
) -> RecorderFactory:
    """Build a RecorderFactory wired to in-memory repos — no DuckDB, no disk."""
    return RecorderFactory(
        ReplayConfig(),
        run_repo=run_repo,
        step_repo=step_repo,
    )


class TestTraceDecorator:
    @pytest.mark.anyio
    async def test_bare_decorator_records_run(
        self,
        run_repo: InMemoryRunRepository,
        step_repo: InMemoryStepRepository,
    ) -> None:
        _set_factory(_make_factory(run_repo, step_repo))

        @trace
        async def agent(x: int) -> int:
            return x * 2

        result = await agent(5)
        assert result == 10

        runs = await run_repo.list_runs()
        assert len(runs) == 1
        assert runs[0].name == "agent"

    @pytest.mark.anyio
    async def test_decorator_with_name_and_tags(
        self,
        run_repo: InMemoryRunRepository,
        step_repo: InMemoryStepRepository,
    ) -> None:
        _set_factory(_make_factory(run_repo, step_repo))

        @trace(name="my-custom-name", tags=["prod"])
        async def agent() -> str:
            return "done"

        await agent()
        runs = await run_repo.list_runs()
        assert runs[0].name == "my-custom-name"
        assert "prod" in runs[0].tags

    @pytest.mark.anyio
    async def test_exceptions_are_reraised(
        self,
        run_repo: InMemoryRunRepository,
        step_repo: InMemoryStepRepository,
    ) -> None:
        _set_factory(_make_factory(run_repo, step_repo))

        @trace
        async def broken_agent() -> None:
            raise ValueError("agent failed")

        with pytest.raises(ValueError, match="agent failed"):
            await broken_agent()

    @pytest.mark.anyio
    async def test_context_var_clean_after_completion(
        self,
        run_repo: InMemoryRunRepository,
        step_repo: InMemoryStepRepository,
    ) -> None:
        _set_factory(_make_factory(run_repo, step_repo))

        @trace
        async def agent() -> None:
            pass

        assert get_recorder() is None
        await agent()
        assert get_recorder() is None

    @pytest.mark.anyio
    async def test_context_var_clean_after_exception(
        self,
        run_repo: InMemoryRunRepository,
        step_repo: InMemoryStepRepository,
    ) -> None:
        _set_factory(_make_factory(run_repo, step_repo))

        @trace
        async def broken() -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await broken()

        assert get_recorder() is None

    @pytest.mark.anyio
    async def test_run_closed_after_completion(
        self,
        run_repo: InMemoryRunRepository,
        step_repo: InMemoryStepRepository,
    ) -> None:
        _set_factory(_make_factory(run_repo, step_repo))

        @trace
        async def agent() -> None:
            pass

        await agent()
        runs = await run_repo.list_runs()
        assert runs[0].ended_at is not None

    @pytest.mark.anyio
    async def test_run_closed_after_exception(
        self,
        run_repo: InMemoryRunRepository,
        step_repo: InMemoryStepRepository,
    ) -> None:
        _set_factory(_make_factory(run_repo, step_repo))

        @trace
        async def broken() -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await broken()

        runs = await run_repo.list_runs()
        assert runs[0].ended_at is not None

    @pytest.mark.anyio
    async def test_record_step_accessible_inside_trace(
        self,
        run_repo: InMemoryRunRepository,
        step_repo: InMemoryStepRepository,
    ) -> None:
        from replay.integrations.raw import record_step

        _set_factory(_make_factory(run_repo, step_repo))

        @trace
        async def agent() -> None:
            await record_step(
                type=StepType.LLM_CALL,
                input={"prompt": "hi"},
                output={"text": "hello"},
            )

        await agent()
        runs = await run_repo.list_runs()
        steps = await step_repo.get_steps(runs[0].id)
        assert len(steps) == 1
        assert steps[0].type == StepType.LLM_CALL

    @pytest.mark.anyio
    async def test_record_step_noop_outside_trace(self) -> None:
        from replay.integrations.raw import record_step

        # Must not raise — silent no-op
        await record_step(type=StepType.LLM_CALL, input={}, output={})

    @pytest.mark.anyio
    async def test_nested_trace_links_parent_run(
        self,
        run_repo: InMemoryRunRepository,
        step_repo: InMemoryStepRepository,
    ) -> None:
        _set_factory(_make_factory(run_repo, step_repo))

        @trace(name="child")
        async def child_agent() -> None:
            pass

        @trace(name="parent")
        async def parent_agent() -> None:
            await child_agent()

        await parent_agent()

        runs = await run_repo.list_runs(limit=10)
        by_name = {r.name: r for r in runs}
        assert "parent" in by_name
        assert "child" in by_name
        assert by_name["child"].parent_run_id == by_name["parent"].id
