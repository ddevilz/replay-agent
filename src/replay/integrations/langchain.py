from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from replay.core.context import get_recorder
from replay.types import StepType


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class ReplayCallbackHandler:
    """LangChain callback handler that records LLM and tool events as Replay Steps.

    Usage:
        from replay.integrations.langchain import ReplayCallbackHandler
        chain = my_chain.with_config(callbacks=[ReplayCallbackHandler()])

    Lazy import: LangChain is only imported inside the methods that need it.
    Raises ImportError with a helpful message if langchain is not installed.
    """

    def __init__(self) -> None:
        self._llm_starts: dict[str, datetime] = {}
        self._tool_starts: dict[str, datetime] = {}
        self._last_messages: dict[str, Any] = {}
        self._verify_langchain()

    @staticmethod
    def _verify_langchain() -> None:
        try:
            import langchain  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "LangChain is not installed. Run: pip install 'replay-agent[langchain]'"
            ) from exc

    # ──────────────────────── LLM events ──────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._llm_starts[str(run_id)] = _now()
        self._last_messages[str(run_id)] = {"prompts": prompts}

    async def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        recorder = get_recorder()
        if recorder is None:
            return

        run_key = str(run_id)
        started_at = self._llm_starts.pop(run_key, _now())
        messages = self._last_messages.pop(run_key, {})

        generation = response.generations[0][0] if response.generations else None
        text = getattr(generation, "text", "") if generation else ""

        llm_output: dict[str, Any] = response.llm_output or {}
        token_usage = llm_output.get("token_usage", {})

        await recorder.add_step(
            type=StepType.LLM_CALL,
            input=messages,
            output={"text": text},
            started_at=started_at,
            ended_at=_now(),
            tokens_in=token_usage.get("prompt_tokens"),
            tokens_out=token_usage.get("completion_tokens"),
            model=llm_output.get("model_name"),
        )

        self._llm_starts.pop(run_key, None)

    # ──────────────────────── Tool events ─────────────────────────────────────

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._tool_starts[str(run_id)] = _now()

    async def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        recorder = get_recorder()
        if recorder is None:
            return

        run_key = str(run_id)
        started_at = self._tool_starts.pop(run_key, _now())

        await recorder.add_step(
            type=StepType.TOOL_CALL,
            input={"tool": name or "unknown"},
            output={"result": output},
            started_at=started_at,
            ended_at=_now(),
        )

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        recorder = get_recorder()
        if recorder is None:
            return

        run_key = str(run_id)
        started_at = self._tool_starts.pop(run_key, _now())

        await recorder.add_step(
            type=StepType.TOOL_CALL,
            input={"tool": name or "unknown"},
            output={},
            started_at=started_at,
            ended_at=_now(),
            error=repr(error),
        )
