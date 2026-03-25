from __future__ import annotations

import pytest

from replay.core.circuit_breaker import CircuitBreaker
from replay.core.recorder import Recorder
from replay.strategies.redaction import NoOpRedaction
from replay.types import StepType
from tests.conftest import InMemoryRunRepository, InMemoryStepRepository


def _make_recorder(
    run_repo: InMemoryRunRepository,
    step_repo: InMemoryStepRepository,
) -> Recorder:
    return Recorder(
        name="test-agent",
        tags=["test"],
        run_repo=run_repo,
        step_repo=step_repo,
        redaction=NoOpRedaction(),
        circuit_breaker=CircuitBreaker(failure_threshold=3),
    )


class TestRecorder:
    @pytest.mark.anyio
    async def test_start_persists_run(
        self,
        run_repo: InMemoryRunRepository,
        step_repo: InMemoryStepRepository,
    ) -> None:
        recorder = _make_recorder(run_repo, step_repo)
        await recorder.start()
        run = await run_repo.get_run(recorder.run_id)
        assert run is not None
        assert run.name == "test-agent"

    @pytest.mark.anyio
    async def test_add_step_increments_index(
        self,
        run_repo: InMemoryRunRepository,
        step_repo: InMemoryStepRepository,
    ) -> None:
        recorder = _make_recorder(run_repo, step_repo)
        await recorder.start()

        await recorder.add_step(type=StepType.LLM_CALL, input={"a": 1}, output={"b": 2})
        await recorder.add_step(type=StepType.TOOL_CALL, input={}, output={})

        steps = await step_repo.get_steps(recorder.run_id)
        assert len(steps) == 2
        assert steps[0].index == 0
        assert steps[1].index == 1

    @pytest.mark.anyio
    async def test_add_step_applies_redaction(
        self,
        run_repo: InMemoryRunRepository,
        step_repo: InMemoryStepRepository,
    ) -> None:
        from replay.strategies.redaction import FieldRedaction

        recorder = Recorder(
            name="test",
            tags=[],
            run_repo=run_repo,
            step_repo=step_repo,
            redaction=FieldRedaction(["api_key"]),
            circuit_breaker=CircuitBreaker(),
        )
        await recorder.start()
        await recorder.add_step(
            type=StepType.LLM_CALL,
            input={"api_key": "sk-secret", "prompt": "hello"},
            output={"text": "world"},
        )

        steps = await step_repo.get_steps(recorder.run_id)
        assert steps[0].input["api_key"] == "***REDACTED***"
        assert steps[0].input["prompt"] == "hello"

    @pytest.mark.anyio
    async def test_add_step_calculates_cost(
        self,
        run_repo: InMemoryRunRepository,
        step_repo: InMemoryStepRepository,
    ) -> None:
        recorder = _make_recorder(run_repo, step_repo)
        await recorder.start()
        await recorder.add_step(
            type=StepType.LLM_CALL,
            input={},
            output={},
            model="gpt-4o",
            tokens_in=1000,
            tokens_out=500,
        )

        steps = await step_repo.get_steps(recorder.run_id)
        assert steps[0].cost_usd is not None
        assert steps[0].cost_usd > 0

    @pytest.mark.anyio
    async def test_finish_sets_ended_at(
        self,
        run_repo: InMemoryRunRepository,
        step_repo: InMemoryStepRepository,
    ) -> None:
        recorder = _make_recorder(run_repo, step_repo)
        await recorder.start()
        await recorder.finish()

        run = await run_repo.get_run(recorder.run_id)
        assert run is not None
        assert run.ended_at is not None

    @pytest.mark.anyio
    async def test_circuit_breaker_open_does_not_raise(
        self,
        run_repo: InMemoryRunRepository,
        step_repo: InMemoryStepRepository,
    ) -> None:
        """A broken recorder must never propagate errors into the agent."""
        from replay.strategies.redaction import NoOpRedaction

        failing_step_repo = InMemoryStepRepository()

        async def _boom(step: object) -> None:
            raise RuntimeError("disk full")

        failing_step_repo.save_step = _boom  # type: ignore[method-assign]

        recorder = Recorder(
            name="test",
            tags=[],
            run_repo=run_repo,
            step_repo=failing_step_repo,
            redaction=NoOpRedaction(),
            circuit_breaker=CircuitBreaker(failure_threshold=1),
        )
        await recorder.start()

        # Should not raise — circuit breaker absorbs the error
        result = await recorder.add_step(type=StepType.LLM_CALL, input={}, output={})
        assert result is None
        assert recorder.circuit_breaker.is_open
