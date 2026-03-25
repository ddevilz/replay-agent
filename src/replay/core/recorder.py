from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from replay.core.circuit_breaker import CircuitBreaker
from replay.storage.repository import RunRepository, StepRepository
from replay.strategies.cost import calculate_cost
from replay.strategies.redaction import RedactionStrategy
from replay.types import Run, Step, StepType


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


def _duration_ms(started: datetime, ended: datetime) -> int:
    return max(0, int((ended - started).total_seconds() * 1000))


class Recorder:
    """Coordinates a single run: creates the Run record, accumulates Steps,
    and closes the run on completion.

    All storage writes go through the CircuitBreaker so that any internal
    failure is absorbed and never surfaces into the user's agent.
    """

    def __init__(
        self,
        *,
        name: str,
        tags: list[str],
        run_repo: RunRepository,
        step_repo: StepRepository,
        redaction: RedactionStrategy,
        circuit_breaker: CircuitBreaker,
        parent_run_id: Optional[str] = None,
        fork_at_step: Optional[int] = None,
    ) -> None:
        self._name = name
        self._tags = tags
        self._run_repo = run_repo
        self._step_repo = step_repo
        self._redaction = redaction
        self._cb = circuit_breaker
        self._parent_run_id = parent_run_id
        self._fork_at_step = fork_at_step

        self._run_id = _new_id()
        self._step_index = 0
        self._run: Optional[Run] = None

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._cb

    async def start(self) -> None:
        """Persist the Run record. Called by the decorator before fn executes."""
        run = Run(
            id=self._run_id,
            name=self._name,
            started_at=_now(),
            tags=self._tags,
            parent_run_id=self._parent_run_id,
            fork_at_step=self._fork_at_step,
        )
        self._run = run
        await self._cb.call(self._run_repo.save_run, run)

    async def add_step(
        self,
        *,
        type: StepType,  # noqa: A002
        input: dict[str, Any],
        output: dict[str, Any],
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
        error: Optional[str] = None,
        model: Optional[str] = None,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        cached_tokens_in: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[Step]:
        """Record a single step. Returns the saved Step, or None if the circuit is open."""
        t_end = ended_at or _now()
        t_start = started_at or t_end

        cost: Optional[float] = None
        if model and tokens_in is not None and tokens_out is not None:
            cost = calculate_cost(
                model,
                tokens_in,
                tokens_out,
                cached_tokens_in=cached_tokens_in or 0,
            )

        step = Step(
            id=_new_id(),
            run_id=self._run_id,
            index=self._step_index,
            type=type,
            started_at=t_start,
            ended_at=t_end,
            duration_ms=_duration_ms(t_start, t_end),
            input=self._redaction.redact(input),
            output=self._redaction.redact(output),
            error=error,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cached_tokens_in=cached_tokens_in,
            cost_usd=cost,
            metadata=metadata or {},
        )

        self._step_index += 1
        await self._cb.call(self._step_repo.save_step, step)
        # Return None when the circuit is open — the step was not durably saved.
        # Callers treat None as "step dropped silently".
        return None if self._cb.is_open else step

    async def finish(self, *, error: Optional[str] = None) -> None:
        """Close the run. Called in the decorator's finally block — always runs."""
        ended_at = _now()
        await self._cb.call(self._run_repo.update_run_ended, self._run_id, ended_at)
