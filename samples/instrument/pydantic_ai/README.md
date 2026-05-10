# PydanticAI instrumentation sample

End-to-end demo of `PydanticAIAdapter` — wraps a one-shot
`Agent("openai:gpt-4o-mini")`, runs `agent.run_sync`, and ships
telemetry to atlas-app via `HttpEventSink`.

## Prerequisites

```bash
pip install 'layerlens[pydantic-ai,providers-openai]'
```

The `pydantic-ai` extra installs `pydantic-ai>=0.0.13,<1.0`
(Python 3.10+).

Required environment:

- `OPENAI_API_KEY` — used by the `"openai:gpt-4o-mini"` model spec.
  If unset the sample exits with code `2`.
- `LAYERLENS_STRATIX_API_KEY` — your LayerLens API key (optional).
- `LAYERLENS_STRATIX_BASE_URL` — atlas-app base URL (optional).

## Run

```bash
uv run python -m samples.instrument.pydantic_ai.main
```

## What this demonstrates

| Component | What it proves (source: `main.py`) |
|---|---|
| `PydanticAIAdapter(capture_config=CaptureConfig.standard())` | Standard capture profile. |
| `adapter.add_sink(HttpEventSink(...))` | Batched HTTP transport. |
| `adapter.connect()` / `adapter.disconnect()` | Full lifecycle. |
| `adapter.instrument_agent(agent)` | Wraps the PydanticAI `Agent`; module docstring states each run emits `agent.input` + `model.invoke` + `agent.output`. |
| `agent.run_sync("What is 2 + 2?")` | One synchronous run; `result.data` is printed. |
| `result.usage()` | When non-`None`, the sample prints request/response/total token counts — proves usage propagation through the wrapper. |

## Expected output

When `OPENAI_API_KEY` is unset:

```text
OPENAI_API_KEY is not set; cannot run sample.
```

Exit code: `2`.

When `pydantic-ai` is not installed:

```text
pydantic-ai not installed. Install with:
    pip install 'layerlens[pydantic-ai,providers-openai]'
```

Exit code: `2`.

When the call succeeds:

```text
Response: <result.data>
Tokens — request: <n>, response: <n>, total: <n>
Telemetry shipped. Check the LayerLens dashboard adapter health page.
```

(The token line is conditional on `result.usage()` returning a
non-`None` value.)

## Multi-tenancy note

This sample does not pass `org_id` to `PydanticAIAdapter`. The
constructor does not yet accept `org_id` — production multi-tenant
wiring lands with the PR #118 adapter-side contract (currently
DRAFT). Once PR #118 merges, pass `org_id` to the adapter so every
emitted event carries it.
