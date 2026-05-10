# OpenAI Agents SDK instrumentation sample

End-to-end demo of `OpenAIAgentsAdapter` — registers a LayerLens trace
processor with the OpenAI Agents SDK, runs `Runner.run_sync` against
a one-shot `Agent`, and ships every SDK span as a LayerLens event.

## Prerequisites

```bash
pip install 'layerlens[openai-agents]' openai-agents
```

> The `openai-agents` extra in `pyproject.toml` pulls `openai>=1.30,<2`
> but the `openai-agents` SDK package itself is not in the extra and
> must be installed explicitly.

Required environment:

- `OPENAI_API_KEY` — used by the underlying OpenAI client. If unset
  the sample exits with code `2`.
- `LAYERLENS_STRATIX_API_KEY` — your LayerLens API key (optional).
- `LAYERLENS_STRATIX_BASE_URL` — atlas-app base URL (optional).

## Run

```bash
uv run python -m samples.instrument.openai_agents.main
```

## What this demonstrates

| Component | What it proves (source: `main.py`) |
|---|---|
| `OpenAIAgentsAdapter(capture_config=CaptureConfig.standard())` | Adapter constructed with the standard capture profile. |
| `adapter.add_sink(HttpEventSink(...))` | Events batched to `/telemetry/spans` (`max_batch=10`, `flush_interval_s=1.0`). |
| `adapter.connect()` / `adapter.disconnect()` | Full lifecycle. |
| `adapter.instrument_runner(None)` | Registers the LayerLens trace processor with the SDK at the global level — every SDK span (agent, model, tool, handoff) emits a LayerLens event (per `main.py` module docstring). |
| `Runner.run_sync(agent, "What is 2 + 2?")` | One synchronous turn against a `gpt-4o-mini` `Agent`; `result.final_output` is printed. |

## Expected output

When `OPENAI_API_KEY` is unset:

```text
OPENAI_API_KEY is not set; cannot run sample.
```

Exit code: `2`.

When `openai-agents` is not installed:

```text
openai-agents not installed. Install with:
    pip install 'layerlens[openai-agents]' openai-agents
```

Exit code: `2`.

When the call succeeds:

```text
Response: <agent final output>
Telemetry shipped. Check the LayerLens dashboard adapter health page.
```

## Multi-tenancy note

This sample does not pass `org_id` to `OpenAIAgentsAdapter`. The
constructor does not yet accept `org_id` — production multi-tenant
wiring lands with the PR #118 adapter-side contract (currently
DRAFT). Once PR #118 merges, pass `org_id` to the adapter so every
emitted event carries it.
