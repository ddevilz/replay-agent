"""Patch openai.resources.chat.AsyncCompletions and Completions.

Wraps both sync and async create() to record one Step per call.
Install by calling install(_factory) once at process startup.
If the openai package is not installed this module is a silent no-op.
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
    _mod = sys.modules.get("openai.resources.chat.completions")
    if _mod is None:
        try:
            import openai.resources.chat.completions as _mod  # type: ignore[no-redef]
        except ImportError:
            return

    _installed = True

    original_async_create = _mod.AsyncCompletions.create
    original_sync_create = _mod.Completions.create

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
        from replay.core.context import get_recorder

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
                import anyio
                from replay.types import StepType

                duration_ms = int((time.monotonic() - started) * 1000)
                step_data = _build_step_data(kwargs, response, error_msg, duration_ms)

                async def _save() -> None:
                    await recorder.add_step(type=StepType.LLM_CALL, **step_data)

                anyio.from_thread.run_sync(anyio.run, _save)  # type: ignore[arg-type]

    _mod.AsyncCompletions.create = _async_create  # type: ignore[method-assign]
    _mod.Completions.create = _sync_create  # type: ignore[method-assign]


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
    finish_reason: Optional[str] = None
    tool_calls: Optional[list[Any]] = None
    correlation_key: Optional[str] = None

    output: dict[str, Any] = {}

    if response is not None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            tokens_in = getattr(usage, "prompt_tokens", None)
            tokens_out = getattr(usage, "completion_tokens", None)

        choices = getattr(response, "choices", None)
        if choices:
            first = choices[0]
            finish_reason = getattr(first, "finish_reason", None)
            msg = getattr(first, "message", None)
            if msg is not None:
                content = getattr(msg, "content", None)
                output["content"] = content
                raw_tool_calls = getattr(msg, "tool_calls", None)
                if raw_tool_calls:
                    tool_calls = [
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in raw_tool_calls
                    ]
                    output["tool_calls"] = tool_calls

        response_id = getattr(response, "id", None)
        if response_id:
            correlation_key = f"openai:{response_id}"

    if finish_reason:
        output["finish_reason"] = finish_reason

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
        "metadata": metadata,
    }
