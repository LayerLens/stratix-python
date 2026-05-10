# LangChain instrumentation sample

End-to-end demo of the LayerLens LangChain callback handler. The sample
builds a single LCEL chain (`prompt | llm`) backed by `ChatOpenAI`, attaches
`LayerLensCallbackHandler` at both the LLM and the chain level, and invokes
it once. Every LangChain LLM/chain callback fires a LayerLens event that
ships to atlas-app via `HttpEventSink`.

## Install

```bash
pip install 'layerlens[langchain,providers-openai]' langchain-openai
```

The `langchain` extra pins `langchain>=0.2,<0.4` and `langchain-core>=0.2,<0.4`.
The `providers-openai` extra adds `openai>=1.30,<2`. The default
`pip install layerlens` does NOT pull either — that's the lazy-import
guarantee tested by `tests/instrument/test_lazy_imports.py`.

## Run

```bash
export OPENAI_API_KEY=sk-...
export LAYERLENS_STRATIX_API_KEY=ll-...   # optional
export LAYERLENS_STRATIX_BASE_URL=...     # optional
python -m samples.instrument.langchain.main
```

If `OPENAI_API_KEY` is unset the sample exits with code 2 and a clear
message (see `main.py:30-32`). If `langchain` / `langchain-openai` is not
installed the sample exits with code 2 and points at the right install
command (see `main.py:37-43`).

## Expected output

```text
Response: <one-line model answer>
Events captured: <N>
Telemetry shipped. Check the LayerLens dashboard adapter health page.
```

`Events captured` is the count returned by
`handler.get_events()` (see `main.py:73`). For this LCEL chain the
LangChain callback protocol fires `on_chain_start` / `on_chain_end` and
`on_llm_start` / `on_llm_end`, which the handler translates into
LayerLens telemetry events before flushing them to the sink.

## What this sample demonstrates

| Component | Where in `main.py` | What it proves |
|---|---|---|
| `HttpEventSink(adapter_name="langchain", path="/telemetry/spans")` | L45-50 | Telemetry transport is configured against the atlas-app spans endpoint with a 10-event batch and 1 s flush interval. |
| `LayerLensCallbackHandler(capture_config=CaptureConfig.standard())` | L52 | Adapter is constructed using the standard capture profile (see `layerlens.instrument.adapters._base.CaptureConfig`). |
| `handler.add_sink(sink)` + `handler.connect()` | L53-54 | The sink is wired to the handler and the adapter lifecycle is brought up to `HEALTHY`. |
| `ChatOpenAI(..., callbacks=[handler])` | L57 | The handler is attached at the LLM level so `on_llm_*` callbacks fire. |
| `chain.invoke({"question": ...}, config={"callbacks": [handler]})` | L66-69 | The handler is also attached at the invocation level so `on_chain_*` callbacks fire for the LCEL pipeline. |
| `handler.get_events()` | L73 | Events captured during the run are inspectable in-process before the sink flushes. |
| `sink.close()` + `handler.disconnect()` | L75-76 | Adapter is torn down cleanly inside a `try/finally`. |

## Production note (multi-tenancy)

This sample does **not** pass `org_id` to `LayerLensCallbackHandler`. In a
multi-tenant deployment every adapter must be scoped to a tenant — pass
`org_id=...` to the handler constructor (or via the `STRATIX` instance) so
emitted telemetry is correctly attributed. The omission here is acceptable
for a local single-tenant smoke test only.
