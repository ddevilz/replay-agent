"""OTEL receiver for Replay.

Catches GenAI spans from any framework emitting opentelemetry-semantic-conventions
gen_ai.* attributes (AG2, AutoGen 0.4, opentelemetry-instrumentation-openai-v2, etc.).

Usage:
    from replay.otel import install_otel_receiver
    install_otel_receiver()  # call once at startup
"""
from __future__ import annotations

from replay.otel.receiver import install_otel_receiver

__all__ = ["install_otel_receiver"]
