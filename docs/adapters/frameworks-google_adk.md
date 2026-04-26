# Google Agent Development Kit framework adapter

`layerlens.instrument.adapters.frameworks.google_adk.GoogleADKAdapter`
instruments [Google ADK](https://github.com/google/adk-python) agents using
the framework's native 6-callback system.

## Install

```bash
pip install 'layerlens[google-adk]'
```

Pulls `google-adk>=0.1,<1.0`. Requires Python 3.10+.

## Quick start

```python
from google.adk.agents import LlmAgent

from layerlens.instrument.adapters.frameworks.google_adk import (
    GoogleADKAdapter,
    instrument_agent,
)
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="google_adk")
adapter = GoogleADKAdapter()
adapter.add_sink(sink)
adapter.connect()

agent = LlmAgent(name="answerer", model="gemini-2.0-flash", instruction="Be concise.")
adapter.instrument_agent(agent)

# Run via the runner of your choice (Runner, AdkApp, etc.)

adapter.disconnect()
sink.close()
```

`instrument_agent(agent)` is the convenience helper.

## What's wrapped

`adapter.instrument_agent(agent)` attaches all six native ADK callbacks:

- `before_agent_callback` → `agent.input` + `environment.config`
- `after_agent_callback` → `agent.output`
- `before_model_callback` → start timer for the model call
- `after_model_callback` → `model.invoke`
- `before_tool_callback` → start timer for the tool call
- `after_tool_callback` → `tool.call`

ADK callbacks are part of the public agent contract. Setting them is the
recommended integration pattern from Google — no monkey-patching is
required, and `disconnect()` simply clears the local timer state. If your
ADK code uses a different agent type (`SequentialAgent`, `ParallelAgent`),
ensure each member agent is instrumented.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `environment.config` | L4a | First `before_agent_callback` per agent. |
| `agent.input` | L1 | Every `before_agent_callback`. |
| `agent.output` | L1 | Every `after_agent_callback`. |
| `model.invoke` | L3 | Every `after_model_callback`. |
| `tool.call` | L5a | Every `after_tool_callback`. |

## ADK specifics

- **Native callback contract**: ADK guarantees that `before_*` is followed
  by exactly one `after_*` per call. Latency is computed using
  thread-local start timestamps.
- **Multimodal Gemini**: when the model produces multimodal output, the
  emitted `model.invoke` payload includes a `content_types` list (e.g.
  `["text", "image"]`).
- **Tool function names**: extracted from the `tool.name` field on the
  `BeforeToolCallback` context — these match the function name registered
  on the agent.
- **Sequential / parallel agents**: a parent `SequentialAgent` calls
  `before_agent_callback` once per child; the adapter records the parent
  agent name in `parent_agent` on each child event.

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# Recommended.
adapter = GoogleADKAdapter(capture_config=CaptureConfig.standard())

# Drop content for compliance.
adapter = GoogleADKAdapter(
    capture_config=CaptureConfig(
        l1_agent_io=True,
        l3_model_metadata=True,
        l5a_tool_calls=True,
        capture_content=False,
    ),
)
```

## BYOK

ADK reads Google AI / Vertex AI credentials from the standard environment
(`GOOGLE_API_KEY` for Google AI Studio, ADC for Vertex). The adapter does
not own those credentials. For platform-managed BYOK see
`docs/adapters/byok.md` (atlas-app M1.B).
