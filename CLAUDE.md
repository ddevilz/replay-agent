# Replay Agent

Time-travel debugger for AI agents. Record every step, scrub back to any point,
fork from any step, share runs as links.

## What This Is

Python SDK + local UI + cloud sync. The core primitive is an immutable Step.
A Run is a sequence of Steps. A Fork is a new Run branching from a parent at a specific Step index.

## Stack

- Python 3.12 (dev), supports 3.9–3.13 (users)
- DuckDB — local storage, zero infra
- anyio — async runtime, cross-version compat
- Pydantic v2 — all data models
- FastAPI — API server
- Typer — CLI
- React + Vite — frontend UI
- uv — package manager

## Repo Layout

```
replay-agent/
├── CLAUDE.md
├── .claude/
│   ├── rules.md
│   ├── commands.md
│   ├── hooks.md
│   ├── skills.md
│   └── agents.md
│   └── features.md
|
├── src/
│   └── replay/
│       ├── __init__.py
│       ├── types.py
│       ├── core/
│       ├── storage/
│       ├── strategies/
│       ├── integrations/
│       └── cli/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
└── pyproject.toml
```

## Key Decisions (do not relitigate)

| Decision | Choice |
|---|---|
| Async runtime | anyio, never raw asyncio |
| Local storage | DuckDB, append-only steps table |
| Data models | Pydantic v2 |
| Step storage | Immutable, append-only |
| Derived state | Computed at read time, never stored |
| Agent safety | Circuit Breaker on every storage write |
| Public API | `replay/__init__.py` only |
| Package manager | uv |

## Positioning

Every competitor (AgentOps, LangSmith, Langfuse) gives you a viewer.
Replay gives you a debugger — branch from step 5, change the input, rerun,
diff the two timelines side by side. That is the gap.

## Full Context

See `.claude/` for rules, commands, hooks, skills, and agents.
