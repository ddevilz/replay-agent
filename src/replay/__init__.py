"""Replay — time-travel debugger for AI agents.

3-line promise:
    import replay

    @replay.trace
    async def my_agent(input: str) -> str: ...

Nothing else is required for basic instrumentation.
"""
from __future__ import annotations

from replay.config import ReplayConfig
from replay.core.context import get_recorder
from replay.core.decorator import _get_factory, _set_factory, trace
from replay.core.factory import RecorderFactory
from replay.core.recorder import Recorder
from replay.core.session import session
from replay.integrations.raw import record_step
from replay.types import Run, RunSummary, Step, StepType

__all__ = [
    # Primary API
    "trace",
    "session",
    "record_step",
    "configure",
    "install",
    # Types
    "Step",
    "StepType",
    "Run",
    "RunSummary",
    # Config
    "ReplayConfig",
    # Introspection
    "get_recorder",
    "Recorder",
]


def install(
    patch_openai: bool = True,
    patch_anthropic: bool = True,
) -> None:
    """Install SDK-level patches for OpenAI and Anthropic clients.

    Call once at process startup, before any LLM calls:

        import replay
        replay.install()

    Providers whose SDK is not installed are silently skipped.
    """
    factory = _get_factory()
    if patch_openai:
        from replay.integrations.sdk_patches.openai_patch import install as _oi
        _oi(factory)
    if patch_anthropic:
        from replay.integrations.sdk_patches.anthropic_patch import install as _an
        _an(factory)


def configure(
    *,
    db_path: object = None,
    redact_pii: bool = False,
    redact_fields: list[str] | None = None,
    circuit_breaker_threshold: int = 3,
    max_step_size_kb: int = 100,
    disabled: bool = False,
    log_level: str = "WARNING",
) -> ReplayConfig:
    """Configure the global Replay SDK settings.

    Must be called before the first @replay.trace invocation if non-default
    settings are needed. Subsequent calls replace the global factory.

        replay.configure(redact_pii=True, db_path="/tmp/runs.db")

    Returns the active ReplayConfig for inspection.
    """
    import logging
    from pathlib import Path

    logging.getLogger("replay").setLevel(log_level)

    kwargs: dict[str, object] = {
        "redact_pii": redact_pii,
        "redact_fields": redact_fields or [],
        "circuit_breaker_threshold": circuit_breaker_threshold,
        "max_step_size_kb": max_step_size_kb,
        "disabled": disabled,
        "log_level": log_level,
    }
    if db_path is not None:
        kwargs["db_path"] = Path(str(db_path))

    config = ReplayConfig(**kwargs)  # type: ignore[arg-type]
    _set_factory(RecorderFactory(config))
    return config
