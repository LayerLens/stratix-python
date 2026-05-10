# Agno instrumentation sample

End-to-end demo of `AgnoAdapter` — wraps a one-shot Agno `Agent`
backed by `OpenAIChat("gpt-4o-mini")`, runs a single `agent.run()`
call, and ships telemetry to atlas-app via `HttpEventSink`.

## Prerequisites

```bash
pip install 'layerlens[agno,providers-openai]'
```

Required environment:

- `OPENAI_API_KEY` — used by `OpenAIChat`. If unset the sample exits
  with code `2`.
- `LAYERLENS_STRATIX_API_KEY` — your LayerLens API key (optional).
- `LAYERLENS_STRATIX_BASE_URL` — atlas-app base URL (optional).

## Run

```bash
uv run python -m samples.instrument.agno.main
```

## What this demonstrates

| Component | What it proves (source: `main.py`) |
|---|---|
| `AgnoAdapter(capture_config=CaptureConfig.standard())` | Adapter constructed with the standard capture profile. |
| `adapter.add_sink(HttpEventSink(...))` | Events batched to `/telemetry/spans` (`max_batch=10`, `flush_interval_s=1.0`). |
| `adapter.connect()` / `adapter.disconnect()` | Full lifecycle: instrument before invocation, restore on exit. |
| `adapter.instrument_agent(agent)` | Wraps `Agent.run` so `agent.input` + `model.invoke` + `agent.output` events fire on each call (per `main.py` module docstring). |
| Response print | The sample prints the `response.content` string returned by `Agent.run("What is 2 + 2?")`. |

## Expected output

When `OPENAI_API_KEY` is unset:

```text
OPENAI_API_KEY is not set; cannot run sample.
```

Exit code: `2`.

When `agno` is not installed:

```text
agno not installed. Install with:
    pip install 'layerlens[agno,providers-openai]'
```

Exit code: `2`.

When the call succeeds the sample prints the model response and a
shipping confirmation:

```text
Response: <model reply>
Telemetry shipped. Check the LayerLens dashboard adapter health page.
```

## Multi-tenancy note

This sample does not pass `org_id` to the adapter constructor. The
`AgnoAdapter` constructor does not yet accept `org_id` — production
multi-tenant wiring lands with the PR #118 adapter-side contract
(currently DRAFT). In the interim, set `org_id` per-event through
`CaptureConfig` overrides or via `HttpEventSink` headers on the
caller side. Once PR #118 merges, pass `org_id="<your-org>"` to
`AgnoAdapter(...)` directly.
