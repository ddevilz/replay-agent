"""Unit tests for the Anthropic SDK patch."""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from replay.config import ReplayConfig
from replay.core.context import reset_recorder, set_recorder
from replay.core.factory import RecorderFactory
from replay.types import StepType
from tests.conftest import InMemoryRunRepository, InMemoryStepRepository


def _make_fake_anthropic_module() -> types.ModuleType:
    fake_mod = types.ModuleType("anthropic.resources.messages")

    class Messages:
        def create(self, **kwargs: Any) -> Any:
            ...

    class AsyncMessages:
        async def create(self, **kwargs: Any) -> Any:
            ...

    fake_mod.Messages = Messages  # type: ignore[attr-defined]
    fake_mod.AsyncMessages = AsyncMessages  # type: ignore[attr-defined]
    return fake_mod


def _make_fake_response(
    response_id: str = "msg_abc123",
    model: str = "claude-sonnet-4-6",
    content: str = "Hello!",
    tokens_in: int = 12,
    tokens_out: int = 8,
    cached_tokens_in: int = 4,
) -> MagicMock:
    response = MagicMock()
    response.id = response_id
    response.model = model
    response.stop_reason = "end_turn"

    usage = MagicMock()
    usage.input_tokens = tokens_in
    usage.output_tokens = tokens_out
    usage.cache_read_input_tokens = cached_tokens_in
    response.usage = usage

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = content
    response.content = [text_block]

    return response


class TestAnthropicPatch:
    def _setup(self) -> tuple[InMemoryRunRepository, InMemoryStepRepository, RecorderFactory]:
        run_repo = InMemoryRunRepository()
        step_repo = InMemoryStepRepository()
        factory = RecorderFactory(ReplayConfig(), run_repo=run_repo, step_repo=step_repo)
        return run_repo, step_repo, factory

    @pytest.mark.anyio
    async def test_async_create_records_step(self) -> None:
        run_repo, step_repo, factory = self._setup()
        recorder = factory.create(name="test-run", tags=[])
        await recorder.start()

        fake_mod = _make_fake_anthropic_module()
        fake_response = _make_fake_response()

        from replay.integrations.sdk_patches import anthropic_patch
        anthropic_patch._installed = False

        async def _mock_create(self: Any, **kwargs: Any) -> Any:
            return fake_response

        fake_mod.AsyncMessages.create = _mock_create  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"anthropic.resources.messages": fake_mod}):
            anthropic_patch.install(factory)

            token = set_recorder(recorder)
            client = fake_mod.AsyncMessages()  # type: ignore[attr-defined]
            result = await client.create(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
            )
            reset_recorder(token)

        anthropic_patch._installed = False

        assert result is fake_response
        steps = await step_repo.get_steps(recorder.run_id)
        assert len(steps) == 1
        step = steps[0]
        assert step.type == StepType.LLM_CALL
        assert step.model == "claude-sonnet-4-6"
        assert step.tokens_in == 12
        assert step.tokens_out == 8
        assert step.cached_tokens_in == 4
        assert step.error is None
        assert step.metadata.get("_correlation_key") == "anthropic:msg_abc123"

    @pytest.mark.anyio
    async def test_async_create_records_error_step(self) -> None:
        run_repo, step_repo, factory = self._setup()
        recorder = factory.create(name="test-run", tags=[])
        await recorder.start()

        fake_mod = _make_fake_anthropic_module()

        from replay.integrations.sdk_patches import anthropic_patch
        anthropic_patch._installed = False

        async def _failing_create(self: Any, **kwargs: Any) -> Any:
            raise RuntimeError("overloaded")

        fake_mod.AsyncMessages.create = _failing_create  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"anthropic.resources.messages": fake_mod}):
            anthropic_patch.install(factory)

            token = set_recorder(recorder)
            with pytest.raises(RuntimeError, match="overloaded"):
                client = fake_mod.AsyncMessages()  # type: ignore[attr-defined]
                await client.create(model="claude-sonnet-4-6", messages=[], max_tokens=100)
            reset_recorder(token)

        anthropic_patch._installed = False

        steps = await step_repo.get_steps(recorder.run_id)
        assert len(steps) == 1
        assert steps[0].error == "overloaded"
