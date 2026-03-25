from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Optional

from replay.core.context import get_recorder
from replay.types import StepType


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _verify_langgraph() -> None:
    try:
        import langgraph  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "LangGraph is not installed. Run: pip install 'replay-agent[langgraph]'"
        ) from exc


class ReplayTracer:
    """LangGraph tracer that records node entry/exit as AGENT_DECISION steps.

    Usage:
        from replay.integrations.langgraph import ReplayTracer
        graph = build_graph().with_config(callbacks=[ReplayTracer()])

    Each node transition is captured with the state snapshot at entry and exit.
    """

    def __init__(self) -> None:
        _verify_langgraph()
        self._node_starts: dict[str, datetime] = {}
        self._node_inputs: dict[str, Any] = {}

    def on_node_start(
        self,
        node_name: str,
        input_state: dict[str, Any],
        run_id: str,
    ) -> None:
        key = f"{run_id}:{node_name}"
        self._node_starts[key] = _now()
        self._node_inputs[key] = input_state

    async def on_node_end(
        self,
        node_name: str,
        output_state: dict[str, Any],
        run_id: str,
        error: Optional[str] = None,
    ) -> None:
        recorder = get_recorder()
        if recorder is None:
            return

        key = f"{run_id}:{node_name}"
        started_at = self._node_starts.pop(key, _now())
        input_state = self._node_inputs.pop(key, {})

        await recorder.add_step(
            type=StepType.AGENT_DECISION,
            input={"node": node_name, "state": input_state},
            output={"state": output_state},
            started_at=started_at,
            ended_at=_now(),
            error=error,
            metadata={"node_name": node_name},
        )


def make_replay_middleware(tracer: Optional[ReplayTracer] = None) -> Callable[..., Any]:
    """Factory for a LangGraph middleware that instruments node execution.

    Returns a middleware function compatible with LangGraph's add_middleware API.
    """
    _verify_langgraph()
    active_tracer = tracer or ReplayTracer()

    async def middleware(
        node_name: str,
        state: dict[str, Any],
        next_fn: Callable[..., Any],
        run_id: str = "",
    ) -> Any:
        active_tracer.on_node_start(node_name, state, run_id)
        result = await next_fn(state)
        await active_tracer.on_node_end(node_name, result or {}, run_id)
        return result

    return middleware
