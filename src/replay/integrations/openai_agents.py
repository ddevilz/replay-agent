"""OpenAI Agents SDK adapter for Replay.

Implements both TracingProcessor and RunHooks interfaces from the agents SDK.
If the agents SDK is not installed, calling install() is a silent no-op.

Usage:
    from replay.integrations.openai_agents import install
    install()  # registers processor globally
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from replay.core.factory import RecorderFactory

# Span type → StepType mapping
_SPAN_TYPE_MAP = {
    "generation": "llm_call",
    "function": "tool_call",
    "agent": "agent_decision",
    "handoff": "agent_decision",
    "mcp_tools": "mcp_call",
    "guardrail": "custom",
}


class ReplayTracingProcessor:
    """Receives span events from the OpenAI Agents SDK and records Steps.

    Register via agents.tracing.add_trace_processor(ReplayTracingProcessor(factory)).
    """

    def __init__(self, factory: "RecorderFactory") -> None:
        self._factory = factory

    def on_trace_start(self, trace: Any) -> None:
        pass

    def on_trace_end(self, trace: Any) -> None:
        pass

    def on_span_start(self, span: Any) -> None:
        pass

    def on_span_end(self, span: Any) -> None:
        import anyio
        from replay.core.context import get_recorder

        recorder = get_recorder()
        if recorder is None:
            return

        span_data = getattr(span, "span_data", None)
        span_type = getattr(span_data, "type", None) or _infer_span_type(span)
        step_type_value = _SPAN_TYPE_MAP.get(str(span_type), "custom")

        from replay.types import StepType
        step_type = StepType(step_type_value)

        input_data = _extract_input(span, span_data)
        output_data = _extract_output(span, span_data)
        error_msg = _extract_error(span)

        started_at = getattr(span, "started_at", None)
        ended_at = getattr(span, "ended_at", None)

        metadata: dict[str, Any] = {}
        agent_name = getattr(span_data, "name", None) or getattr(span, "name", None)
        if agent_name:
            metadata["agent_name"] = agent_name
        span_id = getattr(span, "span_id", None)
        if span_id:
            metadata["span_id"] = span_id
        parent_id = getattr(span, "parent_id", None)
        if parent_id:
            metadata["parent_span_id"] = parent_id

        tokens_in: Optional[int] = None
        tokens_out: Optional[int] = None
        model: Optional[str] = None

        if step_type == StepType.LLM_CALL:
            usage = getattr(span_data, "usage", None)
            if usage:
                tokens_in = (
                    getattr(usage, "input_tokens", None)
                    or getattr(usage, "prompt_tokens", None)
                )
                tokens_out = (
                    getattr(usage, "output_tokens", None)
                    or getattr(usage, "completion_tokens", None)
                )
            model = getattr(span_data, "model", None)

        async def _save() -> None:
            await recorder.add_step(
                type=step_type,
                input=input_data,
                output=output_data,
                started_at=started_at,
                ended_at=ended_at,
                error=error_msg,
                model=model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                metadata=metadata,
            )

        try:
            anyio.from_thread.run_sync(anyio.run, _save)  # type: ignore[arg-type]
        except RuntimeError:
            import asyncio
            asyncio.ensure_future(_save())

    def force_flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class ReplayAgentsHooks:
    """Optional RunHooks integration for richer per-agent metadata.

    Complements ReplayTracingProcessor by providing structured lifecycle events.
    """

    def __init__(self, factory: "RecorderFactory") -> None:
        self._factory = factory

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        pass

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        pass

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        pass

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        pass

    async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
        pass


def install(factory: Optional["RecorderFactory"] = None) -> None:
    """Register ReplayTracingProcessor with the global agents tracing pipeline.

    If factory is None, uses the global factory from replay.core.decorator.
    Silent no-op if the agents SDK is not installed.
    """
    try:
        from agents.tracing import add_trace_processor  # type: ignore[import-not-found]
    except ImportError:
        return

    if factory is None:
        from replay.core.decorator import _get_factory
        factory = _get_factory()

    add_trace_processor(ReplayTracingProcessor(factory))


# ──────────────────────────────────────────────────────────────────────────────
# Span extraction helpers
# ──────────────────────────────────────────────────────────────────────────────


def _infer_span_type(span: Any) -> str:
    name = str(getattr(span, "name", "") or "").lower()
    if "generation" in name or "llm" in name or "chat" in name:
        return "generation"
    if "tool" in name or "function" in name:
        return "function"
    if "handoff" in name:
        return "handoff"
    if "guardrail" in name:
        return "guardrail"
    if "mcp" in name:
        return "mcp_tools"
    return "agent"


def _extract_input(span: Any, span_data: Any) -> dict[str, Any]:
    for attr in ("input", "messages", "prompt"):
        val = getattr(span_data, attr, None)
        if val is not None:
            return {attr: _safe_serialize(val)}
    return {}


def _extract_output(span: Any, span_data: Any) -> dict[str, Any]:
    for attr in ("output", "response", "result"):
        val = getattr(span_data, attr, None)
        if val is not None:
            return {attr: _safe_serialize(val)}
    return {}


def _extract_error(span: Any) -> Optional[str]:
    error = getattr(span, "error", None)
    if error is None:
        return None
    if isinstance(error, str):
        return error or None
    msg = getattr(error, "message", None) or str(error)
    return msg or None


def _safe_serialize(val: Any) -> Any:
    if isinstance(val, (str, int, float, bool, type(None))):
        return val
    if isinstance(val, (list, tuple)):
        return [_safe_serialize(v) for v in val]
    if isinstance(val, dict):
        return {k: _safe_serialize(v) for k, v in val.items()}
    if hasattr(val, "model_dump"):
        return val.model_dump()
    return str(val)
