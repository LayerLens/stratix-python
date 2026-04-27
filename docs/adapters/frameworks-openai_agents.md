# OpenAI Agents SDK framework adapter

`layerlens.instrument.adapters.frameworks.openai_agents.OpenAIAgentsAdapter`
instruments the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
by registering a custom `TracingProcessor` and wrapping `Runner.run` for
execution lifecycle events.

## Install

```bash
pip install 'layerlens[openai-agents]' openai-agents
```

The OpenAI Agents SDK ships as `openai-agents` (separate from the `openai`
client). The `openai-agents` extra here pulls the prerequisite `openai>=1.30`
client; the agents framework itself is installed separately to keep the
optional-deps surface clean.

## Quick start

```python
from agents import Agent, Runner

from layerlens.instrument.adapters.frameworks.openai_agents import (
    OpenAIAgentsAdapter,
    instrument_runner,
)
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="openai_agents")
adapter = OpenAIAgentsAdapter()
adapter.add_sink(sink)
adapter.connect()
adapter.instrument_runner(None)  # registers the global trace processor

agent = Agent(name="answerer", model="gpt-4o-mini", instructions="Be concise.")
result = Runner.run_sync(agent, "What is 2+2?")
print(result.final_output)

adapter.disconnect()
sink.close()
```

## What's wrapped

`adapter.instrument_runner(...)` registers a custom
`agents.tracing.TracingProcessor` via `agents.add_trace_processor()`. The
processor receives every span the SDK produces — agent runs, model calls,
function tools, handoffs, guardrails — and translates them into LayerLens
events.

> **Note**: the OpenAI Agents SDK exposes `add_trace_processor` but no
> matching `remove_trace_processor`. `disconnect()` flips the adapter's
> internal `_connected` flag — the registered processor is still attached
> to the SDK but stops emitting events. To fully remove the processor,
> the SDK process must be restarted.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `environment.config` | L4a | First agent span observed. |
| `agent.input` | L1 | Per agent span start. |
| `agent.output` | L1 | Per agent span end. |
| `agent.action` | L4a | Per `response_span` (model call decision). |
| `agent.handoff` | L4a | Per `handoff_span`. |
| `tool.call` | L5a | Per `function_span`. |
| `model.invoke` | L3 | Per `generation_span` (model call). |
| `policy.violation` | cross-cutting | Per `guardrail_span` that fails. |

## OpenAI Agents specifics

- **Span hierarchy**: each event payload includes `span_id` + `parent_span_id`
  + `trace_id` from the SDK so the platform can reconstruct the agent run
  tree exactly.
- **Handoffs**: the SDK's first-class `handoff` primitive maps cleanly to
  `agent.handoff` with `source_agent` + `target_agent` + `tool_args`
  (when the handoff carries arguments).
- **Guardrails**: input/output guardrails emit `policy.violation` with
  the guardrail name and the rendered reason.
- **Function tools**: tool name and JSON-encoded args/return are captured;
  schemas come from `tool.params_json_schema`.
- **Streaming**: streamed runs (`Runner.run_streamed`) emit one
  consolidated `model.invoke` per generation span on completion.

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# Recommended.
adapter = OpenAIAgentsAdapter(capture_config=CaptureConfig.standard())

# Compliance: drop content but keep span structure.
adapter = OpenAIAgentsAdapter(
    capture_config=CaptureConfig(
        l1_agent_io=True,
        l3_model_metadata=True,
        l5a_tool_calls=True,
        capture_content=False,
    ),
)
```

## BYOK

The OpenAI Agents SDK uses the standard OpenAI client for model calls and
reads `OPENAI_API_KEY` from the environment. The adapter does not own the
key. For platform-managed BYOK see `docs/adapters/byok.md` (atlas-app M1.B).
