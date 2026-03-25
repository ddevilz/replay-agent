from __future__ import annotations

import pytest

from replay.core.circuit_breaker import CircuitBreaker


async def _ok() -> str:
    return "ok"


async def _fail() -> None:
    raise RuntimeError("storage down")


class TestCircuitBreaker:
    @pytest.mark.anyio
    async def test_passes_through_on_success(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        result = await cb.call(_ok)
        assert result == "ok"

    @pytest.mark.anyio
    async def test_resets_failure_count_on_success(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        await cb.call(_fail)
        await cb.call(_fail)
        assert cb.failure_count == 2
        await cb.call(_ok)
        assert cb.failure_count == 0

    @pytest.mark.anyio
    async def test_returns_none_on_failure(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        result = await cb.call(_fail)
        assert result is None

    @pytest.mark.anyio
    async def test_opens_after_threshold_failures(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            await cb.call(_fail)
        assert cb.is_open

    @pytest.mark.anyio
    async def test_open_circuit_skips_all_calls(self) -> None:
        cb = CircuitBreaker(failure_threshold=1)
        await cb.call(_fail)
        assert cb.is_open

        call_count = 0

        async def _side_effect() -> str:
            nonlocal call_count
            call_count += 1
            return "called"

        result = await cb.call(_side_effect)
        assert result is None
        assert call_count == 0  # fn was not called

    @pytest.mark.anyio
    async def test_never_propagates_exceptions(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        # Must not raise — ever
        for _ in range(10):
            result = await cb.call(_fail)
            assert result is None

    @pytest.mark.anyio
    async def test_does_not_open_before_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(4):
            await cb.call(_fail)
        assert not cb.is_open
        assert cb.failure_count == 4
