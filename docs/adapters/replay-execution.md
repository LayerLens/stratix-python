# Adapter Replay Re-execution

This document covers the **factory-based replay** path on
LayerLens framework adapters — the cross-pollination audit
item §2.6 lift that brings the LangChain `execute_replay` pattern
to the eight lighter framework adapters.

It is companion to:

* [`docs/adapters/multi-tenancy.md`](multi-tenancy.md) — the tenant
  binding contract that `ReplayResult` propagates.
* `A:/tmp/adapter-cross-pollination-audit.md` §2.6 — the audit row
  that scopes this lift.

## When to use which replay path

`BaseAdapter` exposes **two** replay entry points:

| Method                           | Caller                | Inputs                                                | Returns                       |
| -------------------------------- | --------------------- | ----------------------------------------------------- | ----------------------------- |
| `execute_replay()`               | LayerLens replay engine | `(inputs, original_trace, request, replay_trace_id)` | A `SerializedTrace`           |
| `execute_replay_via_factory()`   | Adapter SDK / CI tests | `(trace: ReplayableTrace, agent_factory: Callable)`  | A `ReplayResult` (this doc)   |

The **engine** path stays untouched — that is the integration with
the platform replay service that owns trace storage and result routing.

The **factory** path is what this document covers. It is the
self-contained option you reach for when:

* You want to re-run a captured trace through a fresh agent
  *inside the same Python process* (CI, integration tests, debugging).
* You want a divergence report rather than a single pass/fail.
* You want a uniform `ReplayResult` shape across every framework so
  dashboards and `assert` lines do not need adapter-specific branches.

## The pieces

```text
┌─────────────────────────┐  builds   ┌──────────────────┐
│ ReplayExecutor          │──────────>│ ReplayResult     │
│  (shared)               │           │  - trace_id      │
│                         │           │  - source_trace_id│
│  + execute_replay()     │           │  - org_id        │
│                         │           │  - framework     │
│                         │           │  - outputs       │
│                         │           │  - captured_events│
│                         │           │  - divergences[] │
│                         │           │  - duration_ns   │
│                         │           │  - execution_error│
└─────────────┬───────────┘           └──────────────────┘
              │ uses
              v
┌─────────────────────────┐
│ StubInjector            │  (optional, adapter-specific)
│  + build_patches()      │
└─────────────────────────┘
```

* `ReplayExecutor` lives at
  `layerlens.instrument.adapters._base.replay`. It is intentionally
  *narrow* — it does not know how to invoke a framework agent.
* Adapters provide `_invoke_for_replay(agent, inputs, trace)` to
  invoke their framework's run/arun/__call__ entry point.
* Adapters expose `execute_replay_via_factory(trace, agent_factory)`
  as the public surface — typically a 2-line delegate to the
  base helper `_replay_via_executor`.

## Eight wired adapters (cross-pollination audit §2.6)

| Adapter                    | `instrument_*`        | Invocation             |
| -------------------------- | --------------------- | ---------------------- |
| `agno`                     | `instrument_agent`    | `arun` / `run`         |
| `openai_agents`            | `instrument_runner`   | `Runner.run(agent, x)` |
| `llama_index`              | `instrument_workflow` | `arun` / `run`         |
| `google_adk`               | `instrument_agent`    | `run_async` / `run`    |
| `strands`                  | `instrument_agent`    | `__call__` / `invoke`  |
| `pydantic_ai`              | `instrument_agent`    | `run` / `run_sync`     |
| `smolagents`               | `instrument_agent`    | `run(task)`            |
| `ms_agent_framework`       | `instrument_chat`     | `invoke(input=)`       |

LangChain already had its own `execute_replay()` (the original audit
reference) and is not in this lift. Bedrock Agents, browser_use,
embedding, and langfuse are excluded by audit rationale (see §2.6 row
notes — "MAYBE — requires Bedrock-side state, harder",
"N/A — importer / single-agent / no agent concept").

## Honest divergence detection

Per CLAUDE.md ("Honest divergence detection — if replay can't
reproduce exactly, surface it"), `ReplayResult.divergences` is the
*authoritative* report of every event mismatch. The executor never
silently "passes" a divergent replay.

Five divergence kinds are surfaced:

| Kind                  | When it fires                                                  |
| --------------------- | -------------------------------------------------------------- |
| `MISSING_EVENT`       | Original had an event at index N; replay's sequence is shorter |
| `EXTRA_EVENT`         | Replay emitted an event the original did not contain           |
| `EVENT_TYPE_MISMATCH` | Same position, different `event_type`                          |
| `PAYLOAD_MISMATCH`    | Same `event_type`, different meaningful payload field          |
| `EXECUTION_ERROR`     | Framework raised before producing a comparable trace           |

`PAYLOAD_MISMATCH` is deliberately conservative — it compares only
fields whose mismatch genuinely means the agent did something different
(`model`, `provider`, `tool_name`, `agent_name`, `from_agent`,
`to_agent`). Wall-clock fields like `timestamp_ns`, `duration_ns`,
and `run_id` are *expected* to differ between runs and are not
flagged. Flagging them would make every replay "divergent" and
hide real regressions.

`is_exact` reports zero divergences. `succeeded` reports no execution
error. They are orthogonal:

| `succeeded` | `is_exact` | Outcome                                |
| ----------- | ---------- | -------------------------------------- |
| `True`      | `True`     | Perfect reproduction                   |
| `True`      | `False`    | Replay ran but diverged                |
| `False`     | (any)      | Framework crashed during replay        |

## Multi-tenancy

`ReplayResult.org_id` carries the bound tenant from the originating
adapter, set by `BaseAdapter._resolve_org_id` at construction
(see `multi-tenancy.md`). Per-event records inside
`ReplayResult.captured_events` also carry `org_id` because they are
emitted through `_post_emit_success`, which always stamps the field.

A replay started on `adapter_a` (tenant A) cannot leak events into
`adapter_b`'s (tenant B) trace stream — the executor binds to the
adapter at construction and never crosses adapters mid-replay.

## Usage

### Minimal example (agno)

```python
from layerlens.instrument.adapters.frameworks.agno import AgnoAdapter

adapter = AgnoAdapter(org_id="tenant-acme")
adapter.connect()

# 1. Capture an original run.
original_agent = build_my_agno_agent()
adapter.instrument_agent(original_agent)
original_agent.run("Plan a trip to Tokyo")

trace = adapter.serialize_for_replay()

# 2. Replay through a fresh agent built by a factory.
def factory():
    return build_my_agno_agent()  # fresh instance every replay

result = await adapter.execute_replay_via_factory(trace, factory)

assert result.org_id == "tenant-acme"
assert result.framework == "agno"
if not result.is_exact:
    for div in result.divergences:
        print(f"[{div.kind.value}] index={div.index} {div.detail}")
```

### Async factory

The factory may return either an agent instance or an awaitable
resolving to one — the executor inspects the return value and awaits
when needed:

```python
async def async_factory():
    config = await load_config_from_db()
    return AgnoAgent.from_config(config)

result = await adapter.execute_replay_via_factory(trace, async_factory)
```

### Adapter-specific stub injection

For LLM-deterministic replay, supply a `StubInjector` that returns
patches the executor will apply for the duration of the replay run.
The base case (no stubs) works for fixture-based tests where the
agent itself is deterministic.

```python
from layerlens.instrument.adapters._base.replay import (
    ReplayExecutor,
    StubInjector,
)

class MyOpenAIStubs(StubInjector):
    def build_patches(self, adapter, trace):
        # Replace ChatCompletions.create with a deterministic fake.
        return [
            ("openai.resources.chat.completions.Completions.create",
             my_fake_create),
        ]

executor = ReplayExecutor(adapter, stub_injector=MyOpenAIStubs())
result = await executor.execute_replay(trace, factory)
```

Stub teardown is guaranteed even on framework error — patches are
unwound in a `finally` block so a failed replay leaves no global
monkey-patches behind.

## Failure modes (what is *not* swallowed)

| Failure                        | Handling                                                     |
| ------------------------------ | ------------------------------------------------------------ |
| Agent raises mid-execution     | Captured into `result.execution_error`; replay marked failed |
| Agent factory itself raises    | Captured into `result.execution_error`                       |
| Stub teardown raises           | Logged at WARNING; original execution outcome preserved      |
| `org_id` cannot be resolved    | `BaseAdapter.__init__` raises `ValueError` (fail-fast)       |
| Adapter never overrode the method | `NotImplementedError` from `BaseAdapter.execute_replay_via_factory` |

The first two are intentional — collecting them as data lets a
replay-batch consumer aggregate partial results across many traces
without exception-handling boilerplate.
