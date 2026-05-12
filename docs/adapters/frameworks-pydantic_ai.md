# PydanticAI framework adapter

`layerlens.instrument.adapters.frameworks.pydantic_ai.PydanticAIAdapter`
instruments [PydanticAI](https://github.com/pydantic/pydantic-ai) agents by
wrapping `Agent.run()` and `Agent.run_sync()`.

## Install

```bash
pip install 'layerlens[pydantic-ai]'
```

Pulls `pydantic-ai>=0.0.13,<1.0`. Requires Python 3.10+.

## Quick start

```python
from pydantic_ai import Agent

from layerlens.instrument.adapters.frameworks.pydantic_ai import (
    PydanticAIAdapter,
    instrument_agent,
)
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="pydantic_ai")
adapter = PydanticAIAdapter()
adapter.add_sink(sink)
adapter.connect()

agent = Agent("openai:gpt-4o-mini", system_prompt="Be concise.")
adapter.instrument_agent(agent)

result = agent.run_sync("What is 2 + 2?")
print(result.data)

adapter.disconnect()
sink.close()
```

`instrument_agent(agent)` is the convenience helper.

## What's wrapped

`adapter.instrument_agent(agent)` wraps the agent's two entry points:

- `run` — async coroutine. Emits `agent.input` at start, `agent.output` at
  end. Captures intermediate `model.invoke` and `tool.call` events from the
  PydanticAI message history.
- `run_sync` — synchronous wrapper. Same semantics.

`disconnect()` restores both methods to their originals.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `environment.config` | L4a | First wrap of each agent. |
| `agent.input` | L1 | Beginning of every `run` / `run_sync`. |
| `agent.output` | L1 | End of every `run` / `run_sync`. |
| `agent.action` | L4a | Per intermediate model step (multi-step runs). |
| `tool.call` | L5a | Per registered tool invocation. |
| `model.invoke` | L3 | Per LLM call (one per model step). |

The `model.invoke` payload includes the model name (parsed from the
PydanticAI model spec like `openai:gpt-4o-mini`), token usage from
`result.usage()`, and the structured result type if one was declared.

## PydanticAI specifics

- **Structured results**: when an agent declares `result_type=MyModel`, the
  validated Pydantic model is included in `agent.output` (subject to
  `CaptureConfig.capture_content`). Validation errors emit
  `policy.violation`.
- **Model spec parsing**: PydanticAI accepts model spec strings like
  `"openai:gpt-4o-mini"` or `"anthropic:claude-3-5-sonnet"`. The adapter
  splits these into `provider` + `model` for downstream cost lookups.
- **Streaming**: streamed runs (`agent.run_stream`) wrap the async iterator
  and emit a single consolidated `model.invoke` on stream completion. Set
  `stream=False` on the LLM client if you want per-call events.
- **OpenTelemetry compatibility**: PydanticAI also speaks Logfire/OTel.
  The LayerLens adapter and Logfire can run side-by-side; they don't
  conflict because they observe different hooks.

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# Recommended.
adapter = PydanticAIAdapter(capture_config=CaptureConfig.standard())

# Drop content for compliance.
adapter = PydanticAIAdapter(
    capture_config=CaptureConfig(
        l1_agent_io=True,
        l3_model_metadata=True,
        l5a_tool_calls=True,
        capture_content=False,
    ),
)
```

## BYOK

PydanticAI reads provider credentials from the env (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GROQ_API_KEY`, etc.). The adapter does not own them.
For platform-managed BYOK see `docs/adapters/byok.md` (atlas-app M1.B).
