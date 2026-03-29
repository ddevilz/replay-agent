"""Unit tests for the OpenAI SDK patch.

These tests mock out openai's internals so the openai package is not required
to run the test suite. The patch is verified against the mock surface.
"""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from replay.config import ReplayConfig
from replay.core.context import get_recorder, reset_recorder, set_recorder
from replay.core.factory import RecorderFactory
from replay.types import StepType
from tests.conftest import InMemoryRunRepository, InMemoryStepRepository


def _make_fake_openai_module() -> types.ModuleType:
    """Return a minimal fake openai.resources.chat.completions module."""
    fake_completions = types.ModuleType("openai.resources.chat.completions")

    class Completions:
        def create(self, **kwargs: Any) -> Any:
            ...

    class AsyncCompletions:
        async def create(self, **kwargs: Any) -> Any:
            ...

    fake_completions.Completions = Completions  # type: ignore[attr-defined]
    fake_completions.AsyncCompletions = AsyncCompletions  # type: ignore[attr-defined]
    return fake_completions


def _make_fake_response(
    response_id: str = "chatcmpl-abc123",
    model: str = "gpt-4o",
    content: str = "Hello!",
    tokens_in: int = 10,
    tokens_out: int = 5,
) -> MagicMock:
    response = MagicMock()
    response.id = response_id
    response.model = model

    usage = MagicMock()
    usage.prompt_tokens = tokens_in
    usage.completion_tokens = tokens_out
    response.usage = usage

    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None

    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message = msg

    response.choices = [choice]
    return response


class TestOpenAIPatch:
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

        fake_mod = _make_fake_openai_module()
        fake_response = _make_fake_response()

        # Patch and install
        with patch.dict(sys.modules, {"openai.resources.chat.completions": fake_mod}):
            from replay.integrations.sdk_patches import openai_patch
            # Reset state for this test
            openai_patch._installed = False

            async def _mock_create(self: Any, **kwargs: Any) -> Any:
                return fake_response

            fake_mod.AsyncCompletions.create = _mock_create  # type: ignore[attr-defined]

            openai_patch.install(factory)

            token = set_recorder(recorder)
            client = fake_mod.AsyncCompletions()  # type: ignore[attr-defined]
            result = await client.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )
            reset_recorder(token)

            # Reset for other tests
            openai_patch._installed = False

        assert result is fake_response

        steps = await step_repo.get_steps(recorder.run_id)
        assert len(steps) == 1
        step = steps[0]
        assert step.type == StepType.LLM_CALL
        assert step.model == "gpt-4o"
        assert step.tokens_in == 10
        assert step.tokens_out == 5
        assert step.error is None
        assert step.metadata.get("_correlation_key") == "openai:chatcmpl-abc123"
        assert step.input["messages"] == [{"role": "user", "content": "hi"}]

    @pytest.mark.anyio
    async def test_async_create_records_error_step(self) -> None:
        run_repo, step_repo, factory = self._setup()
        recorder = factory.create(name="test-run", tags=[])
        await recorder.start()

        fake_mod = _make_fake_openai_module()

        with patch.dict(sys.modules, {"openai.resources.chat.completions": fake_mod}):
            from replay.integrations.sdk_patches import openai_patch
            openai_patch._installed = False

            async def _failing_create(self: Any, **kwargs: Any) -> Any:
                raise RuntimeError("rate limit exceeded")

            fake_mod.AsyncCompletions.create = _failing_create  # type: ignore[attr-defined]

            openai_patch.install(factory)

            token = set_recorder(recorder)
            with pytest.raises(RuntimeError, match="rate limit exceeded"):
                client = fake_mod.AsyncCompletions()  # type: ignore[attr-defined]
                await client.create(model="gpt-4o", messages=[])
            reset_recorder(token)

            openai_patch._installed = False

        steps = await step_repo.get_steps(recorder.run_id)
        assert len(steps) == 1
        assert steps[0].error == "rate limit exceeded"

    @pytest.mark.anyio
    async def test_no_step_outside_trace_context(self) -> None:
        """Patch must be a no-op when called outside a @trace context."""
        run_repo, step_repo, factory = self._setup()

        fake_mod = _make_fake_openai_module()
        fake_response = _make_fake_response()

        with patch.dict(sys.modules, {"openai.resources.chat.completions": fake_mod}):
            from replay.integrations.sdk_patches import openai_patch
            openai_patch._installed = False

            async def _mock_create(self: Any, **kwargs: Any) -> Any:
                return fake_response

            fake_mod.AsyncCompletions.create = _mock_create  # type: ignore[attr-defined]
            openai_patch.install(factory)

            # No recorder in context
            assert get_recorder() is None
            client = fake_mod.AsyncCompletions()  # type: ignore[attr-defined]
            await client.create(model="gpt-4o", messages=[])

            openai_patch._installed = False

        # Nothing was recorded — no run was ever started
        runs = await run_repo.list_runs()
        assert len(runs) == 0
