# Microsoft Agent Framework instrumentation sample

End-to-end demo of `MSAgentAdapter` — builds a one-shot
`ChatCompletionAgent` backed by `OpenAIChatCompletion("gpt-4o-mini")`,
wraps it via `MSAgentAdapter.instrument_chat`, and runs a single
`agent.invoke` async call.

> **Note on dependencies:** the Microsoft Agent Framework currently
> ships as part of `semantic-kernel`. The `ms-agent-framework` extra
> in `pyproject.toml` resolves to `semantic-kernel>=1.0,<2.0` for
> that reason. The import path the sample uses is
> `semantic_kernel.agents.ChatCompletionAgent`.

## Prerequisites

```bash
pip install 'layerlens[ms-agent-framework,providers-openai]'
```

Required environment:

- `OPENAI_API_KEY` — used by `OpenAIChatCompletion`. If unset the
  sample exits with code `2`.
- `LAYERLENS_STRATIX_API_KEY` — your LayerLens API key (optional).
- `LAYERLENS_STRATIX_BASE_URL` — atlas-app base URL (optional).

## Run

```bash
uv run python -m samples.instrument.ms_agent_framework.main
```

## What this demonstrates

| Component | What it proves (source: `main.py`) |
|---|---|
| `MSAgentAdapter(capture_config=CaptureConfig.standard())` | Standard capture profile. |
| `adapter.add_sink(HttpEventSink(...))` | Batched HTTP transport. |
| `adapter.connect()` / `adapter.disconnect()` | Full lifecycle. |
| `adapter.instrument_chat(agent)` | Wraps the `ChatCompletionAgent`; module docstring states each invocation emits `agent.input` + `model.invoke` + `agent.output`. |
| Async streaming via `async for response in agent.invoke(...)` | Iterates the agent's response stream and collects `response.content` strings. |
| `asyncio.run(_run(agent))` | Demonstrates the sync-driver-around-async-agent pattern. |

## Expected output

When `OPENAI_API_KEY` is unset:

```text
OPENAI_API_KEY is not set; cannot run sample.
```

Exit code: `2`.

When `semantic-kernel` agents are not installed:

```text
semantic-kernel agents not installed. Install with:
    pip install 'layerlens[ms-agent-framework,providers-openai]'
```

Exit code: `2`.

When the call succeeds:

```text
Response: <concatenated agent output>
Telemetry shipped. Check the LayerLens dashboard adapter health page.
```

## Multi-tenancy note

This sample does not pass `org_id` to `MSAgentAdapter`. The
constructor does not yet accept `org_id` — production multi-tenant
wiring lands with the PR #118 adapter-side contract (currently
DRAFT). Once PR #118 merges, pass `org_id` to the adapter so every
emitted event carries it.
