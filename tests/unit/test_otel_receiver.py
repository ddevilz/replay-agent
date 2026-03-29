"""Unit tests for the OTEL span receiver and span mapper.

These tests mock ReadableSpan objects with gen_ai.* attributes and verify
that the correct Steps are emitted. No opentelemetry-sdk required.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from replay.config import ReplayConfig
from replay.core.context import reset_recorder, set_recorder
from replay.core.factory import RecorderFactory
from replay.otel.span_mapper import map_span
from replay.types import StepType
from tests.conftest import InMemoryRunRepository, InMemoryStepRepository


def _make_span(
    op: str = "chat",
    model: str = "gpt-4o",
    tokens_in: int = 10,
    tokens_out: int = 5,
    start_ns: int = 1_000_000_000,
    end_ns: int = 2_000_000_000,
    events: list[Any] | None = None,
    response_id: str | None = None,
    error: bool = False,
) -> MagicMock:
    span = MagicMock()

    attrs: dict[str, Any] = {
        "gen_ai.operation.name": op,
        "gen_ai.request.model": model,
        "gen_ai.usage.input_tokens": tokens_in,
        "gen_ai.usage.output_tokens": tokens_out,
    }
    if response_id:
        attrs["openai.response_id"] = response_id

    span.attributes = attrs
    span.start_time = start_ns
    span.end_time = end_ns
    span.events = events or []
    span.context = None
    span.parent = None

    if error:
        status = MagicMock()
        status.status_code = "StatusCode.ERROR"
        status.description = "model overloaded"
        span.status = status
    else:
        status = MagicMock()
        status.status_code = "StatusCode.OK"
        status.description = None
        span.status = status

    return span


class TestSpanMapper:
    def test_chat_span_maps_to_llm_call(self) -> None:
        span = _make_span(op="chat", model="gpt-4o", tokens_in=10, tokens_out=5)
        step = map_span(span, run_id="run-1", index=0)
        assert step is not None
        assert step.type == StepType.LLM_CALL
        assert step.model == "gpt-4o"
        assert step.tokens_in == 10
        assert step.tokens_out == 5
        assert step.run_id == "run-1"
        assert step.index == 0

    def test_tool_span_maps_to_tool_call(self) -> None:
        span = _make_span(op="tool_call")
        step = map_span(span, run_id="run-1", index=0)
        assert step is not None
        assert step.type == StepType.TOOL_CALL

    def test_agent_span_maps_to_agent_decision(self) -> None:
        span = _make_span(op="agent")
        step = map_span(span, run_id="run-1", index=0)
        assert step is not None
        assert step.type == StepType.AGENT_DECISION

    def test_non_genai_span_returns_none(self) -> None:
        span = MagicMock()
        span.attributes = {"http.method": "GET"}
        span.events = []
        span.context = None
        span.parent = None
        span.start_time = 0
        span.end_time = 1_000_000
        status = MagicMock()
        status.status_code = "StatusCode.OK"
        status.description = None
        span.status = status
        step = map_span(span, run_id="run-1", index=0)
        assert step is None

    def test_error_span_sets_error_field(self) -> None:
        span = _make_span(op="chat", error=True)
        step = map_span(span, run_id="run-1", index=0)
        assert step is not None
        assert step.error == "model overloaded"

    def test_response_id_sets_correlation_key(self) -> None:
        span = _make_span(op="chat", response_id="chatcmpl-xyz")
        step = map_span(span, run_id="run-1", index=0)
        assert step is not None
        assert step.metadata.get("_correlation_key") == "openai:chatcmpl-xyz"

    def test_duration_calculated_from_timestamps(self) -> None:
        span = _make_span(op="chat", start_ns=0, end_ns=500_000_000)  # 500ms
        step = map_span(span, run_id="run-1", index=0)
        assert step is not None
        assert step.duration_ms == 500

    def test_prompt_event_captured_as_input(self) -> None:
        event = MagicMock()
        event.name = "gen_ai.content.prompt"
        event.attributes = {"gen_ai.prompt": "Hello, world!"}
        span = _make_span(op="chat", events=[event])
        step = map_span(span, run_id="run-1", index=0)
        assert step is not None
        assert step.input.get("content") == "Hello, world!"

    def test_completion_event_captured_as_output(self) -> None:
        event = MagicMock()
        event.name = "gen_ai.content.completion"
        event.attributes = {"gen_ai.completion": "Hi there!"}
        span = _make_span(op="chat", events=[event])
        step = map_span(span, run_id="run-1", index=0)
        assert step is not None
        assert step.output.get("content") == "Hi there!"


class TestReplaySpanExporter:
    def _setup(self) -> tuple[InMemoryRunRepository, InMemoryStepRepository, RecorderFactory]:
        run_repo = InMemoryRunRepository()
        step_repo = InMemoryStepRepository()
        factory = RecorderFactory(ReplayConfig(), run_repo=run_repo, step_repo=step_repo)
        return run_repo, step_repo, factory

    @pytest.mark.anyio
    async def test_export_records_step_in_active_recorder(self) -> None:
        run_repo, step_repo, factory = self._setup()
        recorder = factory.create(name="test-run", tags=[])
        await recorder.start()

        from replay.otel.receiver import ReplaySpanExporter
        exporter = ReplaySpanExporter(factory)

        token = set_recorder(recorder)
        span = _make_span(op="chat", model="gpt-4o", tokens_in=8, tokens_out=4)
        exporter.export([span])
        reset_recorder(token)

        # Give asyncio a chance to schedule the future
        import asyncio
        await asyncio.sleep(0)

        steps = await step_repo.get_steps(recorder.run_id)
        assert len(steps) == 1
        assert steps[0].type == StepType.LLM_CALL
        assert steps[0].tokens_in == 8

    @pytest.mark.anyio
    async def test_export_no_op_outside_trace_context(self) -> None:
        run_repo, step_repo, factory = self._setup()

        from replay.otel.receiver import ReplaySpanExporter
        exporter = ReplaySpanExporter(factory)

        span = _make_span(op="chat")
        exporter.export([span])

        runs = await run_repo.list_runs()
        assert len(runs) == 0

    @pytest.mark.anyio
    async def test_suppress_deduplicates_step(self) -> None:
        run_repo, step_repo, factory = self._setup()
        recorder = factory.create(name="test-run", tags=[])
        await recorder.start()

        from replay.otel.receiver import ReplaySpanExporter
        exporter = ReplaySpanExporter(factory)
        exporter.mark_suppress("openai:chatcmpl-dedup")

        token = set_recorder(recorder)
        span = _make_span(op="chat", response_id="chatcmpl-dedup")
        exporter.export([span])
        reset_recorder(token)

        import asyncio
        await asyncio.sleep(0)

        # Step should be suppressed
        steps = await step_repo.get_steps(recorder.run_id)
        assert len(steps) == 0
