from __future__ import annotations

from replay.storage.repository import RunRepository, StepRepository
from replay.types import Run, RunSummary, Step


async def get_full_timeline(run: Run, step_repo: StepRepository) -> list[Step]:
    """Return the complete step sequence for a run, resolving fork ancestry.

    For a root run: returns its own steps in order.
    For a fork: returns parent steps 0..fork_at_step + fork's own steps.

    Storage cost for a fork is only the new steps — the parent history is
    read-only and shared, never copied.
    """
    if run.parent_run_id is None:
        return await step_repo.get_steps(run.id)

    assert run.fork_at_step is not None  # invariant: parent_run_id implies fork_at_step
    parent_steps = await step_repo.get_steps_up_to(run.parent_run_id, run.fork_at_step)
    own_steps = await step_repo.get_steps(run.id)
    return parent_steps + own_steps


async def get_run_summary(run: Run, step_repo: StepRepository) -> RunSummary:
    """Compute RunSummary from the step sequence. Never reads from a summary cache."""
    steps = await get_full_timeline(run, step_repo)

    total_tokens = sum(
        (s.tokens_in or 0) + (s.tokens_out or 0) for s in steps
    )

    costs = [s.cost_usd for s in steps if s.cost_usd is not None]
    total_cost: float | None = sum(costs) if costs else None

    duration_ms = sum(s.duration_ms for s in steps)

    has_error = any(s.error is not None for s in steps)
    if run.ended_at is None:
        computed_status = "running"
    elif has_error:
        computed_status = "failed"
    else:
        computed_status = "completed"

    model_breakdown: dict[str, int] = {}
    for step in steps:
        if step.model:
            model_breakdown[step.model] = model_breakdown.get(step.model, 0) + 1

    return RunSummary(
        run_id=run.id,
        name=run.name,
        started_at=run.started_at,
        ended_at=run.ended_at,
        step_count=len(steps),
        total_tokens=total_tokens,
        total_cost_usd=total_cost,
        duration_ms=duration_ms,
        status=computed_status,
        tags=run.tags,
        model_breakdown=model_breakdown,
    )
