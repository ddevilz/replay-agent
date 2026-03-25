from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from replay.core.context import get_recorder
from replay.types import StepType


async def record_step(
    type: StepType,  # noqa: A002
    input: dict[str, Any],  # noqa: A002
    output: dict[str, Any],
    *,
    started_at: Optional[datetime] = None,
    ended_at: Optional[datetime] = None,
    error: Optional[str] = None,
    model: Optional[str] = None,
    tokens_in: Optional[int] = None,
    tokens_out: Optional[int] = None,
    cached_tokens_in: Optional[int] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Manually record a step inside an active @replay.trace context.

    Silent no-op when called outside a trace context — never raises.
    This is the correct behaviour: utility functions that call record_step
    may or may not be inside a traced run at any given time.

        await record_step(
            type=StepType.LLM_CALL,
            input={"messages": messages},
            output={"text": response.content},
            model="claude-sonnet-4-20250514",
            tokens_in=1200,
            tokens_out=340,
        )
    """
    recorder = get_recorder()
    if recorder is None:
        return

    await recorder.add_step(
        type=type,
        input=input,
        output=output,
        started_at=started_at,
        ended_at=ended_at,
        error=error,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cached_tokens_in=cached_tokens_in,
        metadata=metadata,
    )
