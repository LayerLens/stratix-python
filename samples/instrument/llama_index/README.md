# LlamaIndex instrumentation sample

End-to-end demo of `LlamaIndexAdapter` — registers the LayerLens
event handler with the global LlamaIndex `Dispatcher`, runs a single
LLM `chat` call, and ships events to atlas-app via `HttpEventSink`.

## Prerequisites

```bash
pip install 'layerlens[llama-index,providers-openai]' llama-index-llms-openai
```

> The `llama-index` extra pulls the `llama-index` core meta-package
> but the `llama-index-llms-openai` integration package is separate
> and must be installed explicitly (per `main.py` install hint).

Required environment:

- `OPENAI_API_KEY` — used by `llama_index.llms.openai.OpenAI`. If
  unset the sample exits with code `2`.
- `LAYERLENS_STRATIX_API_KEY` — your LayerLens API key (optional).
- `LAYERLENS_STRATIX_BASE_URL` — atlas-app base URL (optional).

## Run

```bash
uv run python -m samples.instrument.llama_index.main
```

## What this demonstrates

| Component | What it proves (source: `main.py`) |
|---|---|
| `LlamaIndexAdapter(capture_config=CaptureConfig.standard())` | Standard capture profile. |
| `adapter.add_sink(HttpEventSink(...))` | Batched HTTP transport (`max_batch=10`, `flush_interval_s=1.0`). |
| `adapter.connect()` / `adapter.disconnect()` | Full lifecycle. |
| `adapter.instrument_workflow(None)` | Registers the LayerLens event handler at the global Dispatcher level — emits `model.invoke` (and `tool.call` / `agent.*` if produced) per the module docstring. |
| `llm.chat(...)` with `gpt-4o-mini`, `max_tokens=20` | One synchronous chat turn; the assistant message content is printed. |

## Expected output

When `OPENAI_API_KEY` is unset:

```text
OPENAI_API_KEY is not set; cannot run sample.
```

Exit code: `2`.

When `llama-index` is not installed:

```text
llama-index not installed. Install with:
    pip install 'layerlens[llama-index,providers-openai]' llama-index-llms-openai
```

Exit code: `2`.

When the call succeeds:

```text
Response: <assistant content>
Telemetry shipped. Check the LayerLens dashboard adapter health page.
```

## Multi-tenancy note

This sample does not pass `org_id` to `LlamaIndexAdapter`. The
constructor does not yet accept `org_id` — production multi-tenant
wiring lands with the PR #118 adapter-side contract (currently
DRAFT). Once PR #118 merges, pass `org_id` to the adapter so every
emitted event carries it.
