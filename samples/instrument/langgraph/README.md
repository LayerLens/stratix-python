# LayerLens + LangGraph sample

A self-contained 3-node LangGraph state machine wrapped with
`LayerLensLangGraphAdapter`. Designed as a smoke test for the adapter
plumbing and as a worked example of multi-agent tracing — no API keys, no
network round-trips, no external services required.

## What it shows

- **`adapter.wrap_graph(compiled)`** — proxies `invoke` / `ainvoke` and
  emits `environment.config` + `agent.input` + `agent.output` per execution.
- **`agent.state.change`** — emitted whenever the canonical state hash
  changes between snapshots (each of the three nodes mutates state, so
  this fires once per run).
- **`HandoffDetector`** — attached to the adapter so node transitions
  involving an `agent` slot in state emit `agent.handoff`. The standard
  pattern for supervisor / multi-agent graphs.
- **`wrap_llm_for_langgraph(llm, adapter=...)`** — wraps a chat model so
  each invocation also emits `model.invoke` (typed event with token
  usage, model name, provider, latency).
- **Capture configuration** — uses `CaptureConfig.standard()` (L1, L3,
  L4a, L5a, L6 enabled; raw payloads dropped). Swap for
  `CaptureConfig.full()` to keep prompt / response content in the
  emitted events.

## Topology

```
   planner  ->  researcher  ->  writer  ->  END
```

Each node is a pure Python function over the dict-typed graph state. The
`writer` node calls a `MockLLM.invoke(...)` so the sample is offline.

## Run

```bash
pip install 'layerlens[langgraph]'
cd samples/instrument/langgraph
python main.py
```

Expected tail of output:

```
Graph result: {'topic': 'agent observability', 'plan': [...], 'agent': 'writer',
               'research': 'facts about agent observability',
               'summary': 'Drafted summary using cached research.'}
Events emitted: {'agent.input': 1, 'agent.output': 1, 'agent.state.change': 1,
                 'environment.config': 1, 'model.invoke': 1}
```

(Counts may include `agent.handoff` if a `HandoffDetector` is configured
to track the same agent slot the nodes write to.)

## Going to production

Three steps to turn this sample into a real instrumented agent:

1. **Swap the LLM.** Replace `MockLLM()` with `ChatOpenAI(model=...)`
   (or any provider) and pass it through `wrap_llm_for_langgraph(llm,
   adapter=adapter)` so model calls emit typed `model.invoke` events.
2. **Ship telemetry.** Install the transport extra
   (`pip install 'layerlens[transport-http]'`) and the sample will pick
   up `HttpEventSink` automatically — events will POST to
   `LAYERLENS_STRATIX_BASE_URL/telemetry/spans` using
   `LAYERLENS_STRATIX_API_KEY`.
3. **Tighten capture.** Switch to `CaptureConfig.full()` if your
   workspace allows raw prompt / response capture, or define a custom
   `CaptureConfig` to gate specific event layers.

See `docs/adapters/frameworks-langgraph.md` for the full reference.
