"""Patch anthropic.resources.messages.AsyncMessages and Messages.

Wraps both sync and async create() to record one Step per call.
Captures cached_tokens_in from Anthropic's cache_read_input_tokens field.
Install by calling install(_factory) once at process startup.
If the anthropic package is not installed this module is a silent no-op.
"""
from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from replay.core.factory import RecorderFactory

_installed = False


def install(factory: "RecorderFactory") -> None:
    global _installed
    if _installed:
        return

    import sys
    _mod = sys.modules.get("anthropic.resources.messages")
    if _mod is None:
        try:
            import anthropic.resources.messages as _mod  # type: ignore[no-redef]
        except ImportError:
            return

    _installed = True

    original_async_create = _mod.AsyncMessages.create
    original_sync_create = _mod.Messages.create

    @functools.wraps(original_async_create)
    async def _async_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        from replay.core.context import get_recorder
        from replay.types import StepType

        started = time.monotonic()
        recorder = get_recorder()
        error_msg: Optional[str] = None
        response: Any = None

        try:
            response = await original_async_create(self, *args, **kwargs)
            return response
        except Exception as exc:
            error_msg = str(exc) or type(exc).__name__
            raise
        finally:
            if recorder is not None:
                duration_ms = int((time.monotonic() - started) * 1000)
                step_data = _build_step_data(kwargs, response, error_msg, duration_ms)
                await recorder.add_step(type=StepType.LLM_CALL, **step_data)

    @functools.wraps(original_sync_create)
    def _sync_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        import anyio
        from replay.core.context import get_recorder
        from replay.types import StepType

        started = time.monotonic()
        recorder = get_recorder()
        error_msg: Optional[str] = None
        response: Any = None

        try:
            response = original_sync_create(self, *args, **kwargs)
            return response
        except Exception as exc:
            error_msg = str(exc) or type(exc).__name__
            raise
        finally:
            if recorder is not None:
                duration_ms = int((time.monotonic() - started) * 1000)
                step_data = _build_step_data(kwargs, response, error_msg, duration_ms)
                anyio.from_thread.run_sync(
                    lambda: recorder.add_step(type=StepType.LLM_CALL, **step_data)
                )

    _mod.AsyncMessages.create = _async_create  # type: ignore[method-assign]
    _mod.Messages.create = _sync_create  # type: ignore[method-assign]


def _build_step_data(
    kwargs: dict[str, Any],
    response: Any,
    error: Optional[str],
    duration_ms: int,
) -> dict[str, Any]:
    model: Optional[str] = kwargs.get("model")
    messages = kwargs.get("messages", [])

    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cached_tokens_in: Optional[int] = None
    stop_reason: Optional[str] = None
    correlation_key: Optional[str] = None

    output: dict[str, Any] = {}

    if response is not None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            tokens_in = getattr(usage, "input_tokens", None)
            tokens_out = getattr(usage, "output_tokens", None)
            cached_tokens_in = getattr(usage, "cache_read_input_tokens", None)

        stop_reason = getattr(response, "stop_reason", None)
        content = getattr(response, "content", None)
        if content:
            text_blocks = [
                getattr(block, "text", None)
                for block in content
                if getattr(block, "type", None) == "text"
            ]
            if text_blocks:
                output["content"] = text_blocks[0]

        # Use request ID as correlation key for deduplication with OTEL
        response_id = getattr(response, "id", None)
        if response_id:
            correlation_key = f"anthropic:{response_id}"

    if stop_reason:
        output["stop_reason"] = stop_reason

    metadata: dict[str, Any] = {}
    if correlation_key:
        metadata["_correlation_key"] = correlation_key

    return {
        "input": {"messages": messages, "model": model},
        "output": output,
        "error": error,
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cached_tokens_in": cached_tokens_in,
        "metadata": metadata,
    }
