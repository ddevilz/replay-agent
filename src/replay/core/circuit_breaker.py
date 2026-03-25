from __future__ import annotations

import logging
from typing import Any, Callable, Coroutine, TypeVar

logger = logging.getLogger("replay")

_T = TypeVar("_T")


class CircuitBreaker:
    """Guards the user's agent from Replay's internal failures.

    Opens after `failure_threshold` consecutive storage errors.
    Once open, all subsequent calls are silently skipped.
    Logs exactly once when the circuit opens — never on every subsequent call.
    Resets the failure count on any successful call.

    One instance per Recorder — not a global singleton.
    """

    def __init__(self, failure_threshold: int = 3) -> None:
        self._threshold = failure_threshold
        self._failures = 0
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def failure_count(self) -> int:
        return self._failures

    async def call(
        self,
        fn: Callable[..., Coroutine[Any, Any, _T]],
        *args: Any,
        **kwargs: Any,
    ) -> _T | None:
        """Call fn(*args, **kwargs) with circuit-breaker protection.

        Returns None when the circuit is open or when fn raises.
        The caller must treat None as "recording unavailable" — never assume success.
        """
        if self._open:
            return None

        try:
            result = await fn(*args, **kwargs)
            self._failures = 0
            return result
        except Exception as exc:
            self._failures += 1
            if self._failures >= self._threshold:
                self._open = True
                logger.warning(
                    "Replay recorder disabled after %d consecutive failures. "
                    "Last error: %s. Your agent continues normally.",
                    self._threshold,
                    exc,
                )
            return None
