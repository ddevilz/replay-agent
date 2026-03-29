"""ReplaySpanExporter — in-process OTEL SpanExporter.

Registers on the global TracerProvider to intercept GenAI spans from any
framework that emits gen_ai.* attributes (AG2, AutoGen 0.4,
opentelemetry-instrumentation-openai-v2, etc.).

Deduplication: if a span carries openai.response_id or anthropic.request_id,
the corresponding SDK-patch step is suppressed to prevent double-counting.
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    from replay.core.factory import RecorderFactory

_installed = False


def install_otel_receiver(factory: Optional["RecorderFactory"] = None) -> None:
    """Register the Replay OTEL exporter on the global TracerProvider.

    Silent no-op if opentelemetry-sdk is not installed.
    """
    global _installed
    if _installed:
        return

    try:
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # type: ignore[import-not-found]
        from opentelemetry import trace  # type: ignore[import-not-found]
    except ImportError:
        return

    if factory is None:
        from replay.core.decorator import _get_factory
        factory = _get_factory()

    exporter = ReplaySpanExporter(factory)
    provider = trace.get_tracer_provider()

    # If it's already a TracerProvider we can add a processor directly.
    # If it's a ProxyTracerProvider (no SDK configured yet), we set up our own.
    if isinstance(provider, TracerProvider):
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        new_provider = TracerProvider()
        new_provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(new_provider)

    _installed = True


class ReplaySpanExporter:
    """Implements the OTEL SpanExporter interface without inheriting from it.

    Avoids requiring opentelemetry-sdk at import time — we duck-type the
    interface instead. The SDK only checks for .export() and .shutdown().
    """

    def __init__(self, factory: "RecorderFactory") -> None:
        self._factory = factory
        # correlation_key → True: suppress the next SDK-patch step with this key
        self._suppress: dict[str, bool] = {}
        self._lock = threading.Lock()
        # run_id per OTEL trace_id (best-effort: uses active recorder if set)
        self._trace_run_map: dict[int, str] = {}
        # step index per run_id
        self._index_map: dict[str, int] = {}

    def export(self, spans: Sequence[Any]) -> Any:
        from replay.otel.span_mapper import map_span
        from replay.core.context import get_recorder

        try:
            from opentelemetry.sdk.trace.export import SpanExportResult  # type: ignore[import-not-found]
            SUCCESS = SpanExportResult.SUCCESS
        except ImportError:
            SUCCESS = 0

        for span in spans:
            attrs: dict[str, Any] = dict(span.attributes or {})
            op = attrs.get("gen_ai.operation.name")
            if not op:
                continue

            recorder = get_recorder()
            if recorder is None:
                continue

            run_id = recorder.run_id
            with self._lock:
                index = self._index_map.get(run_id, 0)
                self._index_map[run_id] = index + 1

            step = map_span(span, run_id, index)
            if step is None:
                continue

            # Check deduplication
            correlation_key = step.metadata.get("_correlation_key")
            if correlation_key:
                with self._lock:
                    if self._suppress.pop(correlation_key, False):
                        continue

            import anyio
            import asyncio

            async def _save(s: Any = step) -> None:
                await recorder._step_repo.save_step(s)

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(_save())
                else:
                    anyio.run(_save)
            except RuntimeError:
                anyio.run(_save)

        return SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def mark_suppress(self, correlation_key: str) -> None:
        """Tell the exporter to skip the next span with this correlation key."""
        with self._lock:
            self._suppress[correlation_key] = True
