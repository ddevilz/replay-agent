from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from replay.core.context import get_recorder
from replay.types import StepType


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _verify_crewai() -> None:
    try:
        import crewai  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "CrewAI is not installed. Run: pip install 'replay-agent[crewai]'"
        ) from exc


class ReplayCrewCallback:
    """CrewAI callback that records agent step and tool events as Replay Steps.

    Usage:
        from replay.integrations.crewai import ReplayCrewCallback
        crew = Crew(agents=[...], tasks=[...], callbacks=[ReplayCrewCallback()])
    """

    def __init__(self) -> None:
        _verify_crewai()
        self._step_starts: dict[str, datetime] = {}
        self._tool_starts: dict[str, datetime] = {}

    # ──────────────────── Agent step events ───────────────────────────────────

    def on_agent_action(
        self,
        agent_name: str,
        action: str,
        action_input: Any,
        step_id: str = "",
    ) -> None:
        self._step_starts[step_id or agent_name] = _now()

    async def on_agent_finish(
        self,
        agent_name: str,
        output: Any,
        step_id: str = "",
        error: Optional[str] = None,
    ) -> None:
        recorder = get_recorder()
        if recorder is None:
            return

        key = step_id or agent_name
        started_at = self._step_starts.pop(key, _now())

        await recorder.add_step(
            type=StepType.AGENT_DECISION,
            input={"agent": agent_name},
            output={"result": str(output)},
            started_at=started_at,
            ended_at=_now(),
            error=error,
            metadata={"agent_name": agent_name},
        )

    # ──────────────────── Tool events ─────────────────────────────────────────

    def on_tool_use(
        self,
        tool_name: str,
        tool_input: Any,
        step_id: str = "",
    ) -> None:
        self._tool_starts[step_id or tool_name] = _now()

    async def on_tool_result(
        self,
        tool_name: str,
        result: Any,
        step_id: str = "",
        error: Optional[str] = None,
    ) -> None:
        recorder = get_recorder()
        if recorder is None:
            return

        key = step_id or tool_name
        started_at = self._tool_starts.pop(key, _now())

        await recorder.add_step(
            type=StepType.TOOL_CALL,
            input={"tool": tool_name},
            output={"result": str(result)},
            started_at=started_at,
            ended_at=_now(),
            error=error,
            metadata={"tool_name": tool_name},
        )
