from __future__ import annotations

import pytest

from replay.core.circuit_breaker import CircuitBreaker
from replay.core.context import get_recorder
from replay.core.decorator import _set_factory, trace
from replay.core.factory import RecorderFactory
from replay.strategies.redaction import NoOpRedaction
from replay.types import StepType
from tests.conftest import InMemoryRunRepository, InMemoryStepRepository


def _make_factory(
    run_repo: InMemoryRunRepository,
    step_repo: InMemoryStepRepository,
) -> RecorderFactory:
    factory = RecorderFactory.__new__(RecorderFactory)
    factory._config = object()  # type: ignore[assignment]
    factory._run_repo = run_repo
    factory._step_repo = step_repo
    factory._redaction = NoOpRedaction()
    # Patch create to use in-memory repos
    original_create = RecorderFactory.create

    def patched_create(self: RecorderFactory, name: str, tags: list[str], **kwargs: object) -> object:
        from replay.core.recorder import Recorder
        return Recorder(
            name=name,
            tags=tags,
            run_repo=run_repo,
            step_repo=step_repo,
            redaction=NoOpRedaction(),
            circuit_breaker=CircuitBreaker(failure_threshold=3),
        )

    factory.create = patched_create.__get__(factory, RecorderFactory)  # type: ignore[method-assign]
    return factory


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
    async def test_decorator_with_name(
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
    async def test_context_var_reset_after_completion(
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
        assert get_recorder() is None  # must be clean after run

    @pytest.mark.anyio
    async def test_context_var_reset_after_exception(
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

        assert get_recorder() is None  # must be clean even after exception

    @pytest.mark.anyio
    async def test_run_is_closed_after_completion(
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
        steps = await step_repo.get_steps(
            (await run_repo.list_runs())[0].id
        )
        assert len(steps) == 1
        assert steps[0].type == StepType.LLM_CALL

    @pytest.mark.anyio
    async def test_record_step_noop_outside_trace(self) -> None:
        from replay.integrations.raw import record_step

        # Must not raise — silent no-op
        await record_step(
            type=StepType.LLM_CALL,
            input={},
            output={},
        )
