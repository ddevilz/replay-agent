"""Direct SDK patches for OpenAI and Anthropic clients.

Call replay.install() once at startup to activate. If a provider SDK is not
installed, the corresponding patch is a silent no-op.
"""
from __future__ import annotations
