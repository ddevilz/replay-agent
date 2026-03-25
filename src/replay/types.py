from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class StepType(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    MCP_CALL = "mcp_call"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    AGENT_DECISION = "agent_decision"


class Step(BaseModel):
    """Immutable record of a single agent action. Append-only — never mutated after creation."""

    id: str
    run_id: str
    index: int
    type: StepType
    started_at: datetime
    ended_at: datetime
    duration_ms: int
    input: dict[str, Any]
    output: dict[str, Any]
    error: Optional[str] = None
    model: Optional[str] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cached_tokens_in: Optional[int] = None
    cost_usd: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class Run(BaseModel):
    """A named sequence of Steps. Derived fields (total_cost, status) are never stored."""

    id: str
    name: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    tags: list[str] = Field(default_factory=list)
    parent_run_id: Optional[str] = None  # set if this run is a fork
    fork_at_step: Optional[int] = None  # step index the fork branches from

    model_config = {"frozen": True}


class RunSummary(BaseModel):
    """Computed at read time from the step sequence. Never stored."""

    run_id: str
    name: str
    started_at: datetime
    ended_at: Optional[datetime]
    step_count: int
    total_tokens: int
    total_cost_usd: Optional[float]
    duration_ms: int
    status: str  # "running" | "completed" | "failed"
    tags: list[str]
    model_breakdown: dict[str, int]  # model_name -> step count
