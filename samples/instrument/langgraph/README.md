# LangGraph instrumentation sample

End-to-end demo of `LayerLensLangGraphAdapter.wrap_graph` against a
one-node `StateGraph`. The node is a pure Python function (`greet`) that
mutates the graph state — **no LLM provider is involved**, so the sample
runs without `OPENAI_API_KEY` and is safe to use as a wiring smoke test.

## Install

```bash
pip install 'layerlens[langgraph]'
```

The `langgraph` extra pins `langgraph>=0.2,<0.4`. The default
`pip install layerlens` does NOT pull `langgraph` — that's the lazy-import
guarantee tested by `tests/instrument/test_lazy_imports.py`.

## Run

```bash
export LAYERLENS_STRATIX_API_KEY=ll-...   # optional
export LAYERLENS_STRATIX_BASE_URL=...     # optional
python -m samples.instrument.langgraph.main
```

If `langgraph` is not installed the sample exits with code 2 and a clear
install message (see `main.py:31-39`).

## Expected output

```text
Result: {'count': 1, 'messages': ['hi']}
Telemetry shipped. Check the LayerLens dashboard adapter health page.
```

The result reflects the `greet` node's state update: `messages` gets a
single `"hi"` entry and `count` is incremented from `0` to `1` (see
`main.py:52-53`).

## What this sample demonstrates

| Component | Where in `main.py` | What it proves |
|---|---|---|
| `HttpEventSink(adapter_name="langgraph", path="/telemetry/spans")` | L41-46 | Telemetry transport is configured against the atlas-app spans endpoint with a 10-event batch and 1 s flush interval. |
| `LayerLensLangGraphAdapter(capture_config=CaptureConfig.standard())` | L48 | Adapter is constructed using the standard capture profile. |
| `adapter.add_sink(sink)` + `adapter.connect()` | L49-50 | The sink is wired and the adapter lifecycle reaches `HEALTHY` without requiring LangGraph state inspection. |
| Pure-Python `greet` node + `StateGraph(dict)` | L52-59 | The sample requires no LLM — the adapter can be exercised against any LangGraph state machine. |
| `adapter.wrap_graph(compiled)` | L62 | The compiled graph is wrapped; calling `traced.invoke(...)` triggers the adapter's pre/post hooks. |
| `traced.invoke({"count": 0})` | L63 | A single graph execution emits `environment.config` (pre-hook, `lifecycle.py:264-265`), `agent.input` (pre-hook, `lifecycle.py:274-275`), `agent.output` (post-hook, `lifecycle.py:310-311`), and `agent.state.change` (post-hook, `lifecycle.py:323-324`, fires because `greet` mutates state). |
| `sink.close()` + `adapter.disconnect()` | L66-67 | Adapter is torn down cleanly inside a `try/finally`. |

## Production note (multi-tenancy)

This sample does **not** pass `org_id` to `LayerLensLangGraphAdapter`. In a
multi-tenant deployment every adapter must be scoped to a tenant — pass
`org_id=...` to the adapter (or via the `STRATIX` instance) so emitted
telemetry is correctly attributed. The omission here is acceptable for a
local single-tenant smoke test only.
