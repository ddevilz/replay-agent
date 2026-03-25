# Replay

**Time-travel debugger for AI agents.**

Record every step your agent takes. Scrub back to any point. Fork from step 5, change the input, rerun, and diff the two timelines side by side.

Every competitor (AgentOps, LangSmith, Langfuse) gives you a viewer. Replay gives you a debugger.

---

## Install

```bash
pip install replay-agent
```

Requires Python 3.9+. Zero infrastructure — runs entirely on your machine with DuckDB.

---

## Quickstart

```python
import replay

@replay.trace
async def my_agent(query: str) -> str:
    # your agent code here
    return result
```

That's it. Every run is recorded automatically.

---

## Manual Step Recording

For frameworkless agent loops, record steps explicitly:

```python
import replay
from replay.types import StepType

@replay.trace
async def my_agent(messages: list) -> str:
    response = await llm.call(messages)

    await replay.record_step(
        type=StepType.LLM_CALL,
        input={"messages": messages},
        output={"text": response.content},
        model="claude-sonnet-4-20250514",
        tokens_in=response.usage.input_tokens,
        tokens_out=response.usage.output_tokens,
    )

    return response.content
```

`record_step()` is a silent no-op outside a `@replay.trace` context — safe to call anywhere.

---

## Context Manager Form

When you can't use the decorator (e.g. third-party entry points):

```python
async with replay.session("booking-agent") as session:
    result = await agent.run(input)
```

---

## CLI

```bash
# List recent runs
replay ls

# Filter by name or tag
replay ls --name booking-agent --tag prod

# Inspect a specific run
replay show <run-id>

# Full input/output for every step
replay show <run-id> --verbose
```

```
$ replay ls

ID          NAME              STEPS     COST  DURATION  STATUS      WHEN
a1b2c3d4    booking-agent        14   $0.043       23s  completed   2m ago
e5f6g7h8    search-agent          7   $0.012        8s  failed      5m ago
```

---

## Configuration

```python
# Called once before your first @replay.trace
replay.configure(
    redact_pii=True,            # scrub emails, API keys, credit cards
    redact_fields=["api_key"],  # redact specific keys recursively
    db_path="/custom/path.db",  # default: ~/.replay/runs.db
)
```

| Env var | Default | Description |
|---|---|---|
| `REPLAY_DB_PATH` | `~/.replay/runs.db` | Database location |
| `REPLAY_DISABLED` | `0` | Set to `1` to disable all recording |
| `REPLAY_LOG_LEVEL` | `WARNING` | SDK internal log level |

---

## Framework Adapters

### LangChain

```python
from replay.integrations.langchain import ReplayCallbackHandler

chain = my_chain.with_config(callbacks=[ReplayCallbackHandler()])
```

### LangGraph

```python
from replay.integrations.langgraph import ReplayTracer

graph = build_graph().with_config(callbacks=[ReplayTracer()])
```

### CrewAI

```python
from replay.integrations.crewai import ReplayCrewCallback

crew = Crew(agents=[...], tasks=[...], callbacks=[ReplayCrewCallback()])
```

Framework packages are not required at install time — adapters use lazy imports and raise a helpful error if the framework is missing.

---

## Reliability Guarantee

Replay wraps every storage write in a Circuit Breaker. After 3 consecutive failures it stops recording, logs a single warning, and gets out of the way. **Your agent always keeps running.**

```
WARNING replay: Replay recorder disabled after 3 consecutive failures.
               Last error: disk full. Your agent continues normally.
```

Replay is a guest in your system — it never crashes the host.

---

## Redaction

Sensitive data is scrubbed before it reaches the database:

```python
# Regex-based PII scrub (API keys, emails, credit cards)
replay.configure(redact_pii=True)

# Field-level redaction — recursive through nested dicts and lists
replay.configure(redact_fields=["api_key", "user_token", "password"])
```

---

## Cost Tracking

Replay calculates per-step USD cost from model name + token counts. Supported models:

| Model | Input | Output |
|---|---|---|
| claude-sonnet-4-20250514 | $0.003/1K | $0.015/1K |
| claude-opus-4-20250514 | $0.015/1K | $0.075/1K |
| gpt-4o | $0.005/1K | $0.015/1K |
| gpt-4o-mini | $0.00015/1K | $0.0006/1K |
| gemini-1.5-pro | $0.00125/1K | $0.005/1K |

Unknown models show `unknown cost` — never `$0.00`.

---

## Architecture

```
types.py → storage/ → core/ → integrations/ → cli/
```

Strict layer hierarchy. Lower layers never import from higher ones. All derived state (total cost, status, token totals) is computed from the step sequence at read time — never stored. This is what makes fork mode possible.

---

## Development

```bash
# Install with dev dependencies
uv sync --dev

# Run unit tests (fast, no DuckDB)
uv run pytest tests/unit/

# Run integration tests (DuckDB on disk)
uv run pytest tests/integration/

# Full suite
uv run pytest

# Lint + format + type check
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/replay/
```

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 — SDK Core | ✅ Done | `@replay.trace`, `record_step`, DuckDB storage, CLI |
| 2 — Local UI | 🔜 Next | React timeline scrubber, step detail panel, `replay ui` |
| 3 — Fork Engine | 🔜 Planned | Fork from step N, rerun, diff two timelines |
| 4 — Cloud Sync | 🔜 Planned | `replay push`, shareable links |
| 5 — Dashboard | 🔜 Planned | Run history, cost analytics, team features |

---

## License

MIT
