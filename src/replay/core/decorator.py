from __future__ import annotations

import functools
from typing import Any, Callable, Optional, TypeVar, overload

from replay.core.context import get_recorder, reset_recorder, set_recorder
from replay.core.factory import RecorderFactory
from replay.config import ReplayConfig

_F = TypeVar("_F", bound=Callable[..., Any])

# Module-level factory — None until replay.configure() is called.
# Falls back to a default-config factory if configure() was never called.
_global_factory: Optional[RecorderFactory] = None


def _get_factory() -> RecorderFactory:
    global _global_factory
    if _global_factory is None:
        _global_factory = RecorderFactory(ReplayConfig())
    return _global_factory


def _set_factory(factory: RecorderFactory) -> None:
    global _global_factory
    _global_factory = factory


@overload
def trace(func: _F) -> _F: ...


@overload
def trace(
    func: None = None,
    *,
    name: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> Callable[[_F], _F]: ...


def trace(
    func: Optional[_F] = None,
    *,
    name: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> "_F | Callable[[_F], _F]":
    """Wrap an async agent function so every execution is recorded as a Run.

    Bare form:
        @replay.trace
        async def my_agent(input: str) -> str: ...

    With options:
        @replay.trace(name="booking-agent", tags=["prod"])
        async def my_agent(input: str) -> str: ...

    Guarantees:
    - Exceptions are always re-raised — Replay never swallows them.
    - Nested @replay.trace calls create child runs linked to the parent.
    - The ContextVar token is always reset in `finally`, even on exception.
    - recorder.finish() is always called — run is never left open.
    """
    if func is None:
        return functools.partial(trace, name=name, tags=tags)  # type: ignore[return-value]

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        factory = _get_factory()
        run_name = name or func.__name__
        run_tags = tags or []

        parent = get_recorder()
        recorder = factory.create(
            name=run_name,
            tags=run_tags,
            parent_run_id=parent.run_id if parent else None,
        )

        token = set_recorder(recorder)
        try:
            await recorder.start()
            return await func(*args, **kwargs)
        finally:
            # Always runs — closes the run and prevents ContextVar state leak.
            await recorder.finish()
            reset_recorder(token)

    return wrapper  # type: ignore[return-value]
