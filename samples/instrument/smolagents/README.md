# SmolAgents instrumentation sample

End-to-end demo of `SmolAgentsAdapter` — runs **offline** with no
`smolagents` install, no OpenAI key, no network calls. It uses a
duck-typed `_FakeAgent` so the wrapper, lifecycle hooks, and event
emission can be exercised on any developer laptop.

## Run

```bash
python -m samples.instrument.smolagents.main
```

Expected output (event count and order are deterministic):

```text
Agent output: echo: What is 2 + 2?

Emitted 3 event(s):
  -    environment.config  agent=demo-agent
  -           agent.input  agent=demo-agent
  -          agent.output  agent=demo-agent

Replace _FakeAgent with smolagents.CodeAgent and add an
HttpEventSink to ship telemetry to the LayerLens dashboard.
```

## What the sample exercises

| Component | What it proves |
|---|---|
| `SmolAgentsAdapter.connect()` | Adapter reaches `HEALTHY` even when the framework SDK is absent. |
| `SmolAgentsAdapter.instrument_agent(agent)` | `agent.run` is wrapped with the traced shim. |
| Lifecycle hooks | `environment.config`, `agent.input`, `agent.output` are emitted via the recording client. |
| `SmolAgentsAdapter.disconnect()` | `agent.run` is restored to the original. |

## Going to a real run

Swap `_FakeAgent` for a real SmolAgents agent and route events to the
LayerLens dashboard via `HttpEventSink`:

```python
from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool

from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.frameworks.smolagents import SmolAgentsAdapter

sink = HttpEventSink(adapter_name="smolagents")
adapter = SmolAgentsAdapter()
adapter.add_sink(sink)
adapter.connect()

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())
adapter.instrument_agent(agent)
agent.run("What is the weather in Paris today?")

adapter.disconnect()
sink.close()
```

Required env for the live path: `LAYERLENS_STRATIX_API_KEY`,
`LAYERLENS_STRATIX_BASE_URL`, plus whatever credentials your
`smolagents` model wrapper needs.

Install with: `pip install 'layerlens[smolagents]'`.
