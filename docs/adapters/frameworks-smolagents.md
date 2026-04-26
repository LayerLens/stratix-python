# SmolAgents framework adapter

`layerlens.instrument.adapters.frameworks.smolagents.SmolAgentsAdapter`
instruments [SmolAgents](https://github.com/huggingface/smolagents)
(HuggingFace) — `CodeAgent`, `ToolCallingAgent`, and manager → managed
agent topologies — by wrapping `Agent.run()`. SmolAgents has no native
callback system, so the adapter takes the wrapper-pattern path: the
original `run` is preserved on the instance and restored on
`disconnect()`.

## Install

```bash
pip install 'layerlens[smolagents]'
```

Pulls `smolagents>=1.0,<2.0`. Requires Python 3.10+.

## Quick start

```python
from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool

from layerlens.instrument.adapters.frameworks.smolagents import (
    SmolAgentsAdapter,
    instrument_agent,
)
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="smolagents")
adapter = SmolAgentsAdapter()
adapter.add_sink(sink)
adapter.connect()

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=HfApiModel(),
)
adapter.instrument_agent(agent)

result = agent.run("What is the weather in Paris today?")

adapter.disconnect()
sink.close()
```

The `instrument_agent(agent)` module-level convenience function is the
one-liner equivalent — it constructs the adapter, calls `connect()`,
and returns the live adapter so you can register sinks.

For an offline reproduction (no SmolAgents install required) see
`samples/instrument/smolagents/`.

## What's wrapped

`adapter.instrument_agent(agent)` patches the following on each Agent:

- `run` — sync entry point. Emits `agent.input` + `agent.output` and any
  inner `model.invoke` / `tool.call` events you raise via the
  `on_llm_call` / `on_tool_use` hooks.
- For `CodeAgent`, the post-run result drives an `agent.code` event with
  the captured `logs` / `inner_messages` payload.
- Manager agents that own `managed_agents` (dict or list) recurse —
  every managed agent is instrumented exactly once.

`disconnect()` restores all `run` originals and clears internal
bookkeeping.

## Events emitted

| Event                  | Layer          | When                                                                    |
|------------------------|----------------|-------------------------------------------------------------------------|
| `environment.config`   | L4a            | First time an agent is registered. Captures tools, model, system prompt. |
| `agent.input`          | L1             | Beginning of every `run`.                                                |
| `agent.output`         | L1             | End of every `run`. Includes `duration_ns` and any propagated `error`.   |
| `agent.code`           | L2             | After every `CodeAgent.run` whose result carries `logs` / `inner_messages`. |
| `tool.call`            | L5a            | Per `on_tool_use(...)` invocation. Caller raises this from tool code.    |
| `model.invoke`         | L3             | Per `on_llm_call(...)` invocation. Caller raises this from the model layer. |
| `agent.handoff`        | cross-cutting  | Per `on_handoff(...)`, e.g. when a manager delegates to a managed agent. |

`tool.call`, `model.invoke`, and `agent.handoff` are surfaced via
`SmolAgentsAdapter` lifecycle hooks rather than auto-wrapped — call
them explicitly from your tool / model integration code:

```python
adapter.on_tool_use("calculator", tool_input={"x": 2}, tool_output=4)
adapter.on_llm_call(provider="openai", model="gpt-4o-mini", tokens_prompt=150, tokens_completion=42)
adapter.on_handoff(from_agent="planner", to_agent="executor", context="step 3 of plan")
```

This is the same pattern used by the SDK reference adapters where the
upstream framework lacks per-tool / per-LLM-call callbacks.

## SmolAgents specifics

- **Manager agents**: SmolAgents supports manager → managed topologies
  via `managed_agents` on the parent `Agent`. The adapter recurses into
  that attribute (dict OR list) and instruments every member. `agent.handoff`
  is the canonical event for delegation — emit it from your manager's
  delegation hook with `from_agent` / `to_agent` / `context`.
- **CodeAgent vs ToolCallingAgent**: both are wrapped identically. The
  `agent.code` event only fires for `CodeAgent` runs, gated on
  `type(agent).__name__ == "CodeAgent"` to avoid emitting a misleading L2
  event for tool-only agents.
- **Re-instrumentation is idempotent**: `instrument_agent(agent)` keeps a
  per-instance `_originals` map keyed by `id(agent)`. Calling it twice on
  the same agent is a no-op. `environment.config` is emitted once per
  unique `agent_name`.
- **Concurrency**: per-thread `run` start times are tracked under a
  `threading.Lock` so concurrent `run()` calls produce correct
  `duration_ns` per call without inter-thread leakage.

## Capability matrix

| Capability                   | Supported | Notes                                                |
|------------------------------|-----------|------------------------------------------------------|
| `TRACE_TOOLS`                | yes       | Via `on_tool_use(...)` hook.                          |
| `TRACE_MODELS`               | yes       | Via `on_llm_call(...)` hook.                          |
| `TRACE_STATE`                | yes       | `environment.config` captures tools / model / prompt. |
| `TRACE_HANDOFFS`             | yes       | Via `on_handoff(...)` hook for managed-agent delegation. |
| Auto-wrapped tool calls      | no        | SmolAgents has no tool callback; raise events from your tool body. |
| Auto-wrapped model calls     | no        | Likewise — emit `model.invoke` from your model wrapper. |
| Async runs (`arun`)          | no        | SmolAgents 1.x does not expose `arun`. Add when upstream lands it. |

`get_adapter_info().capabilities` reports the four supported
`AdapterCapability` values listed above.

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# Recommended baseline — L1 + L3 + L4a + L5a + L6.
adapter = SmolAgentsAdapter(capture_config=CaptureConfig.standard())

# Heavy: include CodeAgent execution traces and full message payloads.
adapter = SmolAgentsAdapter(capture_config=CaptureConfig.full())

# Custom: keep IO + tool-calls but drop content (PII-sensitive deployments).
adapter = SmolAgentsAdapter(
    capture_config=CaptureConfig(
        l1_agent_io=True,
        l5a_tool_calls=True,
        capture_content=False,
    ),
)
```

When `capture_content=False`, `system_prompt` is not included in
`environment.config`, `messages` are not included in `model.invoke`,
and the `agent.handoff` `context_preview` field is set to `None` —
but the SHA-256 `context_hash` is always retained for correlation.

## Version compatibility

| Component              | Supported range                           |
|------------------------|-------------------------------------------|
| `smolagents`           | `>=1.0,<2.0` (extra: `layerlens[smolagents]`) |
| Python                 | 3.10+                                     |
| Pydantic               | v1 OR v2 (adapter declares `V1_OR_V2` via `requires_pydantic`) |
| `layerlens` core       | matches the SDK release that ships this adapter |

The adapter never imports SmolAgents at module-import time —
`smolagents` is imported lazily inside `connect()` and the missing-import
case is logged at DEBUG, never raised. This keeps `import layerlens` cheap
and lets the manifest emitter introspect the class without the runtime
SDK present.

## BYOK

SmolAgents model wrappers (`HfApiModel`, `LiteLLMModel`, etc.) read their
own credentials. The adapter does not own them. For platform-managed BYOK
key resolution see `docs/adapters/byok.md` (atlas-app M1.B).

## Backward compatibility

`SmolAgentsAdapter` was previously named `STRATIXSmolAgentsAdapter`
under the legacy STRATIX brand. The old name still imports for one
deprecation cycle:

```python
from layerlens.instrument.adapters.frameworks.smolagents import (
    STRATIXSmolAgentsAdapter,  # deprecated — use SmolAgentsAdapter
)
```

Importing the legacy alias raises `DeprecationWarning`. The alias will
be removed in a future major release; migrate to `SmolAgentsAdapter`.
