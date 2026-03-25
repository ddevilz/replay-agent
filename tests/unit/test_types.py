from __future__ import annotations

from datetime import datetime, timezone

import pytest

from replay.types import Run, Step, StepType


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class TestStep:
    def test_frozen(self) -> None:
        step = Step(
            id="s1",
            run_id="r1",
            index=0,
            type=StepType.LLM_CALL,
            started_at=_now(),
            ended_at=_now(),
            duration_ms=100,
            input={"prompt": "hello"},
            output={"text": "world"},
        )
        with pytest.raises(Exception):
            step.index = 99  # type: ignore[misc]

    def test_optional_fields_default_to_none(self) -> None:
        step = Step(
            id="s1",
            run_id="r1",
            index=0,
            type=StepType.TOOL_CALL,
            started_at=_now(),
            ended_at=_now(),
            duration_ms=50,
            input={},
            output={},
        )
        assert step.error is None
        assert step.model is None
        assert step.tokens_in is None
        assert step.cost_usd is None

    def test_metadata_defaults_to_empty_dict(self) -> None:
        step = Step(
            id="s1",
            run_id="r1",
            index=0,
            type=StepType.MCP_CALL,
            started_at=_now(),
            ended_at=_now(),
            duration_ms=10,
            input={},
            output={},
        )
        assert step.metadata == {}


class TestRun:
    def test_fork_fields_default_to_none(self) -> None:
        run = Run(id="r1", name="test", started_at=_now())
        assert run.parent_run_id is None
        assert run.fork_at_step is None

    def test_tags_default_to_empty(self) -> None:
        run = Run(id="r1", name="test", started_at=_now())
        assert run.tags == []

    def test_fork_fields_preserved(self) -> None:
        run = Run(
            id="fork1",
            name="fork",
            started_at=_now(),
            parent_run_id="parent1",
            fork_at_step=3,
        )
        assert run.parent_run_id == "parent1"
        assert run.fork_at_step == 3
