from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from replay.types import Run, Step


class RunRepository(ABC):
    """Abstract interface for persisting and querying Runs."""

    @abstractmethod
    async def save_run(self, run: Run) -> None: ...

    @abstractmethod
    async def get_run(self, run_id: str) -> Optional[Run]: ...

    @abstractmethod
    async def update_run_ended(self, run_id: str, ended_at: object) -> None:
        """Mark the run as finished. ended_at must be a timezone-aware datetime."""
        ...

    @abstractmethod
    async def list_runs(
        self,
        limit: int = 20,
        name: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> list[Run]: ...


class StepRepository(ABC):
    """Abstract interface for persisting and querying Steps.

    The steps table is INSERT-only. There is no update path.
    """

    @abstractmethod
    async def save_step(self, step: Step) -> None: ...

    @abstractmethod
    async def get_steps(self, run_id: str) -> list[Step]: ...

    @abstractmethod
    async def get_step(self, run_id: str, index: int) -> Optional[Step]: ...

    @abstractmethod
    async def get_steps_up_to(self, run_id: str, max_index: int) -> list[Step]:
        """Return all steps for run_id where step.index <= max_index.

        Used by the fork reader to reconstruct the parent's history up to the
        fork point without loading the full run.
        """
        ...
