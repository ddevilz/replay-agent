from __future__ import annotations

from typing import Optional


# USD per 1 000 tokens — update this table when pricing changes.
# Never hardcode rates inline elsewhere in the codebase.
COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    # Claude
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-opus-4-20250514": {"input": 0.015, "output": 0.075},
    "claude-haiku-4-5-20251001": {"input": 0.0008, "output": 0.004},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    # OpenAI
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    # Google
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
}


def calculate_cost(
    model: str,
    tokens_in: int,
    tokens_out: int,
    cached_tokens_in: int = 0,
) -> Optional[float]:
    """Return the USD cost for a single LLM call, or None for unknown models.

    Returns None (not 0.0) for unrecognised model names.
    Returning 0.0 would imply a free call; None correctly signals "unknown cost".
    Cached input tokens are billed at half the standard input rate.
    """
    rates = COST_PER_1K_TOKENS.get(model)
    if rates is None:
        return None

    billable_input = tokens_in - cached_tokens_in
    cached_cost = cached_tokens_in / 1000 * rates["input"] * 0.5
    input_cost = billable_input / 1000 * rates["input"]
    output_cost = tokens_out / 1000 * rates["output"]

    return input_cost + cached_cost + output_cost
