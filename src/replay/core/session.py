from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from replay.core.context import get_recorder, reset_recorder, set_recorder
from replay.core.decorator import _get_factory
from replay.core.recorder import Recorder


@asynccontextmanager
async def session(
    name: str,
    tags: Optional[list[str]] = None,
) -> AsyncIterator[Recorder]:
    """Async context manager alternative to @replay.trace.

    Use when the decorator form cannot be applied (e.g. third-party entry points).

        async with replay.session("booking-agent") as sess:
            result = await agent.run(input)
            # steps are recorded via record_step() inside agent.run()

    The context manager handles start, finish, and ContextVar cleanup
    with the same guarantees as @replay.trace.
    """
    factory = _get_factory()
    parent = get_recorder()

    recorder = factory.create(
        name=name,
        tags=tags or [],
        parent_run_id=parent.run_id if parent else None,
    )

    token = set_recorder(recorder)
    try:
        await recorder.start()
        yield recorder
        await recorder.finish()
    except Exception as exc:
        await recorder.finish(error=repr(exc))
        raise
    finally:
        reset_recorder(token)
