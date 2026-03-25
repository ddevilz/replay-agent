from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any


# Patterns applied recursively to string values
_SENSITIVE_PATTERNS: dict[str, re.Pattern[str]] = {
    "openai_key": re.compile(r"sk-[a-zA-Z0-9]{32,}"),
    "anthropic_key": re.compile(r"sk-ant-[a-zA-Z0-9\-]{32,}"),
    "aws_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "email": re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    "credit_card": re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),
}

_REDACTED = "***REDACTED***"


class RedactionStrategy(ABC):
    """Base class for all redaction strategies.

    Applied before every storage write. Implementors must be stateless and
    deterministic — the same input always produces the same output.
    """

    @abstractmethod
    def redact(self, data: dict[str, Any]) -> dict[str, Any]: ...


class NoOpRedaction(RedactionStrategy):
    """Passthrough — no redaction applied. Default for local development."""

    def redact(self, data: dict[str, Any]) -> dict[str, Any]:
        return data


class FieldRedaction(RedactionStrategy):
    """Redact specific named keys anywhere in the nested structure."""

    def __init__(self, fields: list[str]) -> None:
        self._fields = frozenset(fields)

    def redact(self, data: dict[str, Any]) -> dict[str, Any]:
        return _traverse_fields(data, self._fields)


class PIIRedaction(RedactionStrategy):
    """Regex-based scrub of API keys, emails, and credit card numbers.

    Applied recursively to all string values in the data structure.
    """

    def redact(self, data: dict[str, Any]) -> dict[str, Any]:
        return _traverse_pii(data)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — pure functions, no side effects
# ──────────────────────────────────────────────────────────────────────────────


def _traverse_fields(obj: Any, fields: frozenset[str]) -> Any:
    if isinstance(obj, dict):
        return {
            k: _REDACTED if k in fields else _traverse_fields(v, fields)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_traverse_fields(item, fields) for item in obj]
    return obj


def _traverse_pii(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _traverse_pii(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_traverse_pii(item) for item in obj]
    if isinstance(obj, str):
        return _scrub_string(obj)
    return obj


def _scrub_string(value: str) -> str:
    for name, pattern in _SENSITIVE_PATTERNS.items():
        value = pattern.sub(f"***{name.upper()}***", value)
    return value
