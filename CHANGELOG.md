# Changelog

## 0.1.0 — 2026-03-29

Initial release.

### Added
- `@replay.trace` decorator for async agent functions
- `replay.session()` context manager for manual run scoping
- `replay.record_step()` for manual instrumentation
- `replay.install()` to patch OpenAI and Anthropic SDKs at startup
- LangChain, LangGraph, CrewAI framework adapters
- OpenAI Agents SDK adapter (`ReplayTracingProcessor`, `ReplayAgentsHooks`)
- DuckDB local storage — zero infrastructure required
- `replay ls` and `replay show` CLI commands
- Circuit breaker — Replay never crashes your agent
- PII redaction and field-level redaction
- Cost tracking for Claude, GPT-4o, Gemini models
- OTEL span receiver for GenAI instrumentation frameworks
