"""Map OTEL GenAI spans to Replay Step fields.

Only spans with a gen_ai.operation.name attribute are processed.
All other spans are ignored — Replay is not a general-purpose OTEL collector.
"""
from __future__ import annotations

from typing import Any, Optional

from replay.types import Step, StepType

# gen_ai.operation.name → StepType
_OP_MAP: dict[str, StepType] = {
    "chat": StepType.LLM_CALL,
    "text_completion": StepType.LLM_CALL,
    "embeddings": StepType.LLM_CALL,
    "tool_call": StepType.TOOL_CALL,
    "tool": StepType.TOOL_CALL,
    "agent": StepType.AGENT_DECISION,
    "mcp": StepType.MCP_CALL,
}


def map_span(span: Any, run_id: str, index: int) -> Optional[Step]:
    """Convert a ReadableSpan to a Replay Step.

    Returns None if the span is not a GenAI span or cannot be mapped.
    """
    from datetime import datetime, timezone

    attrs: dict[str, Any] = dict(span.attributes or {})
    op = attrs.get("gen_ai.operation.name")
    if not op:
        return None

    step_type = _OP_MAP.get(str(op).lower(), StepType.CUSTOM)

    # Timing — OTEL uses nanoseconds since epoch
    start_ns = getattr(span, "start_time", None)
    end_ns = getattr(span, "end_time", None)

    def _ns_to_dt(ns: Optional[int]) -> datetime:
        if ns is None:
            return datetime.now(tz=timezone.utc)
        return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)

    started_at = _ns_to_dt(start_ns)
    ended_at = _ns_to_dt(end_ns)
    duration_ms = max(0, int(((end_ns or 0) - (start_ns or 0)) / 1e6))

    model: Optional[str] = attrs.get("gen_ai.request.model") or attrs.get("gen_ai.response.model")
    tokens_in: Optional[int] = _int_attr(attrs, "gen_ai.usage.input_tokens")
    tokens_out: Optional[int] = _int_attr(attrs, "gen_ai.usage.output_tokens")

    # Build input/output from span events (OTEL GenAI semantic conventions)
    input_data: dict[str, Any] = {}
    output_data: dict[str, Any] = {}
    error_msg: Optional[str] = None

    events = list(getattr(span, "events", None) or [])
    for event in events:
        event_name = getattr(event, "name", "")
        event_attrs: dict[str, Any] = dict(getattr(event, "attributes", None) or {})
        if event_name == "gen_ai.content.prompt":
            input_data["content"] = event_attrs.get("gen_ai.prompt")
        elif event_name == "gen_ai.content.completion":
            output_data["content"] = event_attrs.get("gen_ai.completion")

    # Tool / agent name from attributes
    tool_name = attrs.get("gen_ai.tool.name")
    if tool_name:
        input_data["tool_name"] = tool_name
    agent_name = attrs.get("gen_ai.agent.name")
    if agent_name:
        input_data["agent_name"] = agent_name

    # Error from span status
    status = getattr(span, "status", None)
    if status is not None:
        status_code = getattr(status, "status_code", None)
        # StatusCode.ERROR == 2 in opentelemetry-sdk
        if str(status_code) in ("StatusCode.ERROR", "ERROR", "2"):
            error_msg = getattr(status, "description", None) or "error"

    # Correlation key for deduplication with SDK patches
    metadata: dict[str, Any] = {}
    openai_response_id = attrs.get("openai.response_id") or attrs.get("gen_ai.response.id")
    if openai_response_id:
        metadata["_correlation_key"] = f"openai:{openai_response_id}"
    anthropic_request_id = attrs.get("anthropic.request_id")
    if anthropic_request_id:
        metadata["_correlation_key"] = f"anthropic:{anthropic_request_id}"

    span_id = getattr(span, "context", None)
    if span_id:
        span_id_val = getattr(span_id, "span_id", None)
        if span_id_val:
            metadata["otel_span_id"] = format(span_id_val, "016x")
    parent = getattr(span, "parent", None)
    if parent:
        parent_span_id = getattr(parent, "span_id", None)
        if parent_span_id:
            metadata["otel_parent_span_id"] = format(parent_span_id, "016x")

    import uuid
    step_id = str(uuid.uuid4())

    return Step(
        id=step_id,
        run_id=run_id,
        index=index,
        type=step_type,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=duration_ms,
        input=input_data,
        output=output_data,
        error=error_msg,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        metadata=metadata,
    )


def _int_attr(attrs: dict[str, Any], key: str) -> Optional[int]:
    val = attrs.get(key)
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None
