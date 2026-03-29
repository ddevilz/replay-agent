from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


_DEFAULT_DB_PATH = Path.home() / ".replay" / "runs.db"
_DEFAULT_MAX_STEP_SIZE_KB = 100


class ReplayConfig(BaseModel):
    """All runtime configuration for the Replay SDK.

    Constructed by replay.configure() or built from environment variables.
    Immutable after construction — mutating config mid-run leads to undefined behaviour.
    """

    db_path: Path = Field(default_factory=lambda: _DEFAULT_DB_PATH)
    max_step_size_kb: int = Field(default=_DEFAULT_MAX_STEP_SIZE_KB, gt=0)
    circuit_breaker_threshold: int = Field(default=3, gt=0)

    # Redaction
    redact_pii: bool = False
    redact_fields: list[str] = Field(default_factory=list)

    # Observability
    disabled: bool = False  # REPLAY_DISABLED=1 — complete no-op when True
    log_level: str = "WARNING"

    model_config = {"frozen": True}

    @field_validator("db_path", mode="before")
    @classmethod
    def expand_path(cls, v: object) -> Path:
        return Path(str(v)).expanduser()
